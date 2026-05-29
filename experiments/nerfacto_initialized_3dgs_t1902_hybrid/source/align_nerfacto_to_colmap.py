"""
Align a raw Nerfstudio point cloud to COLMAP metric space.

Background
----------
nerfacto_depth_to_ply.py (and ns-export pointcloud) output 3-D points in
Nerfstudio's *normalized* world space.  This space differs from the original
COLMAP metric space by a uniform scale and a translation:

    x_colmap = x_nerfstudio / total_scale + translation

total_scale = dataparser_scale * extra_scale_factor
  - dataparser_scale  : the "scale" field in dataparser_transforms.json,
                        auto-computed by Nerfstudio when fitting camera poses
                        to a unit sphere.
  - extra_scale_factor: an empirical factor that accounts for additional
                        internal Nerfstudio normalization not reflected in
                        dataparser_transforms.json.  For the t1902 wheat scene
                        this factor is 1.836 (so total_scale = 2.46 * 1.836
                        = 4.516).  It may differ for other scenes.

Translation can be supplied manually or derived by centroid-matching the
scaled Nerfstudio cloud to a reference COLMAP cloud.  Note: centroid matching
works well when both clouds cover roughly the same region; it may be inaccurate
if the Nerfstudio cloud captures only the plant while COLMAP covers the full
scene setup.

Verified values for timestep 1902
----------------------------------
    --scale 4.516 --translation 0.507 -0.518 -0.550

Usage examples
--------------
# Reproduce the t1902 aligned cloud exactly:
python align_nerfacto_to_colmap.py \\
    --nerfacto-ply  3dgs_project/data/nerfacto_3163.ply \\
    --out           /tmp/nerfacto_aligned.ply \\
    --scale         4.516 \\
    --translation   0.507 -0.518 -0.550

# Auto-derive scale from dataparser_transforms.json + centroid match to COLMAP:
python align_nerfacto_to_colmap.py \\
    --nerfacto-ply           nerfacto_raw.ply \\
    --out                    nerfacto_aligned.ply \\
    --dataparser-transforms  path/to/dataparser_transforms.json \\
    --extra-scale            1.836 \\
    --colmap-ply             colmap_points3D.ply
"""

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_xyz_rgb(path: Path):
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = v.data.dtype.names or ()
    xyz = np.vstack([v["x"], v["y"], v["z"]]).T.astype(np.float64)
    if all(c in names for c in ("red", "green", "blue")):
        rgb = np.vstack([v["red"], v["green"], v["blue"]]).T.astype(np.uint8)
    else:
        rgb = np.full((len(xyz), 3), 180, dtype=np.uint8)
    return xyz, rgb


def write_wheat3dgs_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(
        len(xyz),
        dtype=[
            ("x", "f8"), ("y", "f8"), ("z", "f8"),
            ("nx", "f8"), ("ny", "f8"), ("nz", "f8"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    for i, field in enumerate(("x", "y", "z")):
        vertices[field] = xyz[:, i]
    for field in ("nx", "ny", "nz"):
        vertices[field] = 0.0
    for i, field in enumerate(("red", "green", "blue")):
        vertices[field] = rgb[:, i]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def cloud_stats(label: str, xyz: np.ndarray) -> None:
    center = (xyz.min(0) + xyz.max(0)) / 2
    extent = xyz.max(0) - xyz.min(0)
    print(f"  {label:30s}  n={len(xyz):6d}  "
          f"center=[{center[0]:+.4f} {center[1]:+.4f} {center[2]:+.4f}]  "
          f"extent=[{extent[0]:.4f} {extent[1]:.4f} {extent[2]:.4f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--nerfacto-ply", required=True, type=Path,
                        help="Raw nerfacto PLY in Nerfstudio world space")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output path for the aligned PLY (Wheat-3DGS format)")

    # --- Scale (mutually exclusive: explicit or derived from dataparser) ---
    scale_grp = parser.add_mutually_exclusive_group(required=True)
    scale_grp.add_argument("--scale", type=float,
                           help="Total scale divisor (e.g. 4.516 for t1902)")
    scale_grp.add_argument("--dataparser-transforms", type=Path,
                           help="dataparser_transforms.json from the nerfacto run directory")

    parser.add_argument("--extra-scale", type=float, default=1.836,
                        help="Empirical factor multiplied with the dataparser scale "
                             "(default 1.836, verified for t1902 wheat scene)")

    # --- Translation (optional; falls back to centroid matching) ---
    parser.add_argument("--translation", type=float, nargs=3,
                        metavar=("TX", "TY", "TZ"),
                        help="Explicit translation vector added after scaling.  "
                             "If omitted, derived by centroid-matching to --colmap-ply.")
    parser.add_argument("--colmap-ply", type=Path,
                        help="COLMAP points3D.ply used as reference for centroid matching "
                             "or for reporting comparison stats")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Determine scale
    # ------------------------------------------------------------------
    if args.scale is not None:
        total_scale = args.scale
        print(f"Scale (explicit):  {total_scale}")
    else:
        with open(args.dataparser_transforms) as f:
            dt = json.load(f)
        dp_scale = float(dt["scale"])
        total_scale = dp_scale * args.extra_scale
        print(f"Scale (derived):   {dp_scale:.6f} (dataparser)  "
              f"× {args.extra_scale} (extra)  =  {total_scale:.6f}")

    # ------------------------------------------------------------------
    # 2. Load input clouds
    # ------------------------------------------------------------------
    nf_xyz, nf_rgb = read_xyz_rgb(args.nerfacto_ply)
    colmap_xyz = None
    if args.colmap_ply is not None:
        colmap_xyz, _ = read_xyz_rgb(args.colmap_ply)

    print("\nInput statistics:")
    cloud_stats("nerfacto (raw)", nf_xyz)
    if colmap_xyz is not None:
        cloud_stats("colmap (reference)", colmap_xyz)

    # ------------------------------------------------------------------
    # 3. Apply scale
    # ------------------------------------------------------------------
    scaled = nf_xyz / total_scale

    # ------------------------------------------------------------------
    # 4. Determine translation
    # ------------------------------------------------------------------
    if args.translation is not None:
        translation = np.array(args.translation, dtype=np.float64)
        print(f"\nTranslation (explicit):  [{translation[0]:+.5f}  "
              f"{translation[1]:+.5f}  {translation[2]:+.5f}]")
    elif colmap_xyz is not None:
        colmap_centroid = colmap_xyz.mean(0)
        scaled_centroid = scaled.mean(0)
        translation = colmap_centroid - scaled_centroid
        print(f"\nTranslation (centroid match to COLMAP):  "
              f"[{translation[0]:+.5f}  {translation[1]:+.5f}  {translation[2]:+.5f}]")
        print("  Note: centroid matching may be inaccurate if the two clouds cover "
              "different regions (e.g. plant-only vs full scene).")
    else:
        translation = np.zeros(3)
        print("\nNo translation source supplied — using zero translation.")
        print("  Provide --translation or --colmap-ply for a non-zero shift.")

    # ------------------------------------------------------------------
    # 5. Apply translation and write output
    # ------------------------------------------------------------------
    aligned = scaled + translation

    print("\nOutput statistics:")
    cloud_stats("aligned (output)", aligned)
    if colmap_xyz is not None:
        cloud_stats("colmap (reference)", colmap_xyz)

    write_wheat3dgs_ply(args.out, aligned, nf_rgb)
    print(f"\nSaved → {args.out}")

    # ------------------------------------------------------------------
    # 6. Quick verification: extent ratio should be uniform
    # ------------------------------------------------------------------
    raw_ext = nf_xyz.max(0) - nf_xyz.min(0)
    aln_ext = aligned.max(0) - aligned.min(0)
    ratios = raw_ext / np.where(aln_ext > 0, aln_ext, 1)
    print(f"\nExtent ratios per axis (raw / aligned): "
          f"[{ratios[0]:.4f}  {ratios[1]:.4f}  {ratios[2]:.4f}]")
    if ratios.max() / ratios.min() > 1.01:
        print("  WARNING: ratios differ by >1% — the scale is not uniform. "
              "Check your --scale value.")
    else:
        print("  OK: uniform scale confirmed.")


if __name__ == "__main__":
    main()
