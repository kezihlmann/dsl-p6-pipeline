from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def read_xyz_rgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ply = PlyData.read(str(path))
    vertices = ply["vertex"]
    names = vertices.data.dtype.names or ()

    xyz = np.vstack(
        [
            np.asarray(vertices["x"], dtype=np.float64),
            np.asarray(vertices["y"], dtype=np.float64),
            np.asarray(vertices["z"], dtype=np.float64),
        ]
    ).T

    if all(channel in names for channel in ("red", "green", "blue")):
        rgb = np.vstack(
            [
                np.asarray(vertices["red"], dtype=np.uint8),
                np.asarray(vertices["green"], dtype=np.uint8),
                np.asarray(vertices["blue"], dtype=np.uint8),
            ]
        ).T
    else:
        rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
        rgb[:, 1] = 180

    return xyz, rgb


def write_wheat3dgs_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    vertices = np.empty(
        len(xyz),
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("nx", "f8"),
            ("ny", "f8"),
            ("nz", "f8"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["nx"] = 0.0
    vertices["ny"] = 0.0
    vertices["nz"] = 0.0
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]

    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the timestep 1902 hybrid COLMAP+Nerfacto 3DGS initializer."
    )
    parser.add_argument("--colmap-ply", required=True, type=Path)
    parser.add_argument("--nerfacto-ply", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--voxel",
        type=float,
        default=0.005,
        help="Voxel size used for rounded-coordinate duplicate removal.",
    )
    args = parser.parse_args()

    colmap_xyz, colmap_rgb = read_xyz_rgb(args.colmap_ply)
    nerfacto_xyz, nerfacto_rgb = read_xyz_rgb(args.nerfacto_ply)

    print("COLMAP points:", len(colmap_xyz))
    print("Nerfacto points:", len(nerfacto_xyz))

    if len(colmap_xyz) == 0:
        raise RuntimeError("COLMAP input has 0 points")
    if len(nerfacto_xyz) == 0:
        raise RuntimeError("Nerfacto input has 0 points")
    if np.isnan(colmap_xyz).any() or np.isnan(nerfacto_xyz).any():
        raise RuntimeError("NaNs found in input point clouds")

    xyz = np.vstack([colmap_xyz, nerfacto_xyz])
    rgb = np.vstack([colmap_rgb, nerfacto_rgb])

    key = np.round(xyz / args.voxel).astype(np.int64)
    _, keep_idx = np.unique(key, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)

    xyz = xyz[keep_idx]
    rgb = rgb[keep_idx]

    write_wheat3dgs_ply(args.out, xyz, rgb)

    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    print("Saved:", args.out)
    print("Hybrid points:", len(xyz))
    print("center:", (mins + maxs) / 2)
    print("extent:", maxs - mins)


if __name__ == "__main__":
    main()
