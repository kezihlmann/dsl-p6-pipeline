from pathlib import Path
import argparse
import torch
import numpy as np

C0 = 0.28209479177387814

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def collect_tensors(obj, prefix=""):
    out = []
    if torch.is_tensor(obj):
        out.append((prefix, obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(collect_tensors(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.extend(collect_tensors(v, f"{prefix}.{i}" if prefix else str(i)))
    return out

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
    ap.add_argument("--min-opacity", type=float, default=0.0)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    if isinstance(ckpt, (list, tuple)) and len(ckpt) == 2:
        payload, iteration = ckpt
        print("Checkpoint iteration:", iteration)
    else:
        payload = ckpt

    tensors = collect_tensors(payload)

    print("Tensor candidates:")
    for name, t in tensors:
        if t.ndim > 0:
            print(f"{name:35s} shape={tuple(t.shape)}")

    # In this checkpoint, tensor 1 is xyz, tensor 2 is features_dc, tensor 6 is opacity.
    # Also keep fallback logic in case names/order differ.
    xyz_t = None
    fdc_t = None
    opacity_t = None

    for name, t in tensors:
        if t.ndim == 2 and t.shape[1] == 3 and t.shape[0] > 100 and xyz_t is None:
            xyz_t = t
            print("Using xyz:", name, tuple(t.shape))
            break

    if xyz_t is None:
        raise RuntimeError("Could not find xyz tensor.")

    n = xyz_t.shape[0]

    for name, t in tensors:
        if t.ndim == 3 and t.shape[0] == n and t.shape[1] == 1 and t.shape[2] == 3:
            fdc_t = t
            print("Using features_dc:", name, tuple(t.shape))
            break

    # Prefer the first Nx1 tensor after xyz/features/scaling/rotation candidates.
    # For your printed checkpoint, this is tensor 6.
    for name, t in tensors:
        if t.ndim == 2 and t.shape[0] == n and t.shape[1] == 1:
            if name == "6" or opacity_t is None:
                opacity_t = t
                print("Using opacity candidate:", name, tuple(t.shape))
                if name == "6":
                    break

    xyz = xyz_t.detach().numpy().astype(np.float32)

    if fdc_t is not None:
        fdc = fdc_t.detach().numpy().reshape(n, 3).astype(np.float32)
        rgb = np.clip(fdc * C0 + 0.5, 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)
    else:
        rgb = np.full((n, 3), 180, dtype=np.uint8)

    if opacity_t is not None:
        opacity = sigmoid(opacity_t.detach().numpy().reshape(-1).astype(np.float32))
    else:
        opacity = np.ones(n, dtype=np.float32)

    if args.min_opacity > 0:
        keep = opacity >= args.min_opacity
        print(f"Opacity filter >= {args.min_opacity}: keeping {keep.sum()} / {len(keep)}")
        xyz = xyz[keep]
        rgb = rgb[keep]
        opacity = opacity[keep]

    write_ply(args.out, xyz, rgb, opacity)
    print("Saved:", args.out)
    print("Vertices:", len(xyz))

if __name__ == "__main__":
    main()
