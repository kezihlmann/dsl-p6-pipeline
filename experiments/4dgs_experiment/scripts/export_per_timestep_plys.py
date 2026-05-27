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
    ap.add_argument("--source", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--window", type=float, default=0.15)
    ap.add_argument("--scale", type=float, default=100.0)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    payload, iteration = ckpt
    print("Checkpoint iteration:", iteration)

    xyz = payload[1].detach().cpu().numpy().astype(np.float32)
    features_dc = payload[2].detach().cpu().numpy().reshape(len(xyz), 3).astype(np.float32)
    opacity_raw = payload[6].detach().cpu().numpy().reshape(-1).astype(np.float32)
    time_raw = payload[13].detach().cpu().numpy().reshape(-1).astype(np.float32)

    opacity = sigmoid(opacity_raw)
    gtime = sigmoid(time_raw)

    rgb = np.clip(features_dc * C0 + 0.5, 0.0, 1.0)
    rgb = (rgb * 255).astype(np.uint8)

    center = xyz.mean(axis=0, keepdims=True)
    xyz_scaled = (xyz - center) * args.scale

    print("xyz:", xyz.shape)
    print("time after sigmoid: min", gtime.min(), "max", gtime.max(), "mean", gtime.mean())

    source = Path(args.source)
    timestep_dirs = sorted(source.glob("timestep_*_data"))
    times = np.linspace(0.0, 1.0, len(timestep_dirs))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, (tdir, tval) in enumerate(zip(timestep_dirs, times)):
        dt = np.abs(gtime - tval)
        keep = dt <= args.window

        # fallback: if too few points, keep closest 300
        if keep.sum() < 120:
            k = min(300, len(dt))
            idx = np.argsort(dt)[:k]
            keep = np.zeros(len(dt), dtype=bool)
            keep[idx] = True

        out = outdir / f"{i:02d}_{tdir.name}_active_centers_scaled.ply"
        write_ply(out, xyz_scaled[keep], rgb[keep], opacity[keep])
        print(f"{tdir.name}: t={tval:.3f}, points={keep.sum()}, saved={out}")

if __name__ == "__main__":
    main()
