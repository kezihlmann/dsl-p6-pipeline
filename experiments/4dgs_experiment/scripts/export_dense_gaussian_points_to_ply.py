from pathlib import Path
import argparse
import torch
import numpy as np

C0 = 0.28209479177387814

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def write_ply(path, xyz, rgb, opacity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(xyz)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float opacity\n")
        f.write("end_header\n")

        for p, c, a in zip(xyz, rgb, opacity):
            f.write(
                f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} {float(a):.6f}\n"
            )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples-per-gaussian", type=int, default=30)
    ap.add_argument("--min-opacity", type=float, default=0.01)
    ap.add_argument("--jitter-scale", type=float, default=1.0)
    ap.add_argument("--view-scale", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    payload, iteration = ckpt
    print("Checkpoint iteration:", iteration)

    # Based on your checkpoint printout:
    # 1 xyz, 2 features_dc, 4 raw scaling, 6 raw opacity
    xyz = payload[1].detach().cpu().numpy().astype(np.float32)
    features_dc = payload[2].detach().cpu().numpy().reshape(len(xyz), 3).astype(np.float32)
    raw_scaling = payload[4].detach().cpu().numpy().astype(np.float32)
    raw_opacity = payload[6].detach().cpu().numpy().reshape(-1).astype(np.float32)

    opacity = sigmoid(raw_opacity)

    rgb = np.clip(features_dc * C0 + 0.5, 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)

    keep = opacity >= args.min_opacity
    xyz = xyz[keep]
    rgb = rgb[keep]
    opacity = opacity[keep]
    raw_scaling = raw_scaling[keep]

    print("Kept Gaussians:", len(xyz))

    # 3DGS scaling parameters are usually stored in log-space.
    scales = np.exp(raw_scaling)

    # Avoid extreme sampled blobs.
    scales = np.clip(scales, 1e-5, 0.02)

    all_pts = []
    all_rgb = []
    all_op = []

    # Include original centers too.
    all_pts.append(xyz)
    all_rgb.append(rgb)
    all_op.append(opacity)

    S = args.samples_per_gaussian

    for i in range(len(xyz)):
        noise = rng.normal(size=(S, 3)).astype(np.float32)
        pts = xyz[i:i+1] + noise * scales[i:i+1] * args.jitter_scale
        all_pts.append(pts)
        all_rgb.append(np.repeat(rgb[i:i+1], S, axis=0))
        all_op.append(np.repeat(opacity[i:i+1], S, axis=0))

    dense_xyz = np.concatenate(all_pts, axis=0)
    dense_rgb = np.concatenate(all_rgb, axis=0)
    dense_op = np.concatenate(all_op, axis=0)

    # Center and scale for easier viewing.
    center = dense_xyz.mean(axis=0, keepdims=True)
    dense_xyz = (dense_xyz - center) * args.view_scale

    write_ply(args.out, dense_xyz, dense_rgb, dense_op)

    print("Saved:", args.out)
    print("Vertices:", len(dense_xyz))

if __name__ == "__main__":
    main()
