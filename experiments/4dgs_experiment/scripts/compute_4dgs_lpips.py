from pathlib import Path
from PIL import Image
import numpy as np
import torch
import lpips
import csv
import argparse

def load_rgb(path):
    arr = np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0
    return arr

def load_mask(path, size):
    m = Image.open(path).convert("L").resize(size, Image.NEAREST)
    return (np.asarray(m) > 127).astype(np.float32)

def to_lpips_tensor(img):
    # img: H,W,3 in [0,1] -> torch 1,3,H,W in [-1,1]
    t = torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    return t * 2.0 - 1.0

def find_mask(source_root, stem):
    # stem example: 000_timestep_0004_data_GX010075_20250910_193302_1520
    parts = stem.split("_")
    timestep = "_".join(parts[1:4])  # timestep_0004_data
    cam = parts[4]                  # GX010075
    mask_dir = Path(source_root) / timestep / "masks"

    candidates = sorted(mask_dir.glob(f"{cam}*_mask*.png"))
    if not candidates:
        candidates = sorted(mask_dir.glob(f"{cam}*.png"))
    if not candidates:
        raise FileNotFoundError(f"No mask found for {stem} in {mask_dir}")
    return candidates[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="lpips_4dgs.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()

    rows = []
    for split in ["test", "train"]:
        split_dir = Path(args.eval_root) / split
        for pred_path in sorted(split_dir.glob("*_pred.png")):
            stem = pred_path.name.replace("_pred.png", "")
            gt_path = split_dir / f"{stem}_gt.png"
            if not gt_path.exists():
                print("missing gt:", gt_path)
                continue

            pred = load_rgb(pred_path)
            gt = load_rgb(gt_path)

            # Full-image LPIPS
            with torch.no_grad():
                full_lpips = loss_fn(
                    to_lpips_tensor(pred).to(device),
                    to_lpips_tensor(gt).to(device)
                ).item()

            # Plant-masked LPIPS: black out background in both gt and pred
            mask_path = find_mask(args.source, stem)
            mask = load_mask(mask_path, (pred.shape[1], pred.shape[0]))[..., None]
            pred_m = pred * mask
            gt_m = gt * mask

            with torch.no_grad():
                plant_lpips = loss_fn(
                    to_lpips_tensor(pred_m).to(device),
                    to_lpips_tensor(gt_m).to(device)
                ).item()

            rows.append({
                "split": split,
                "stem": stem,
                "full_lpips": full_lpips,
                "plant_masked_lpips": plant_lpips,
                "mask_path": str(mask_path),
            })

            print(split, stem, "full LPIPS=", round(full_lpips, 4),
                  "plant-masked LPIPS=", round(plant_lpips, 4))

    out_path = Path(args.eval_root) / args.out
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved:", out_path)

    for key in ["full_lpips", "plant_masked_lpips"]:
        vals = np.array([r[key] for r in rows], dtype=np.float32)
        n = len(vals)
        ci90 = 1.833 * vals.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        print(f"{key}: mean={vals.mean():.4f}, std={vals.std(ddof=1):.4f}, 90% CI=±{ci90:.4f}, n={n}")

if __name__ == "__main__":
    main()
