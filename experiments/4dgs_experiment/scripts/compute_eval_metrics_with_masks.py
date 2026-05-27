from pathlib import Path
import argparse
import csv
import math
import re
import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

THRESHOLDS = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

def load_gray(path):
    return Image.open(path).convert("L")

def psnr_from_mse(mse):
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)

def bbox_from_mask(mask, pad=20):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    y0 = max(0, ys.min() - pad)
    y1 = min(h, ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(w, xs.max() + pad + 1)
    return y0, y1, x0, x1

def parse_name(stem):
    # Example:
    # 000_timestep_2052_data_GX010075_20250912_160302_2054
    if re.match(r"^\d+_", stem):
        stem2 = stem.split("_", 1)[1]
    else:
        stem2 = stem

    m = re.match(r"^(timestep_\d+_data)_(.+)$", stem2)
    if not m:
        raise ValueError(f"Could not parse timestep/image from {stem}")
    return m.group(1), m.group(2)

def find_mask(source, stem, target_size):
    tdir, image_stem = parse_name(stem)
    mask_dir = Path(source) / tdir / "masks"

    candidates = [
        mask_dir / f"{image_stem}_mask_sam3.png",
        mask_dir / f"{image_stem}_mask.png",
        mask_dir / f"{image_stem}_mask_ground_truth.png",
        mask_dir / f"{image_stem}.png",
    ]

    mask_path = next((p for p in candidates if p.exists()), None)
    if mask_path is None:
        raise FileNotFoundError(f"No mask found for {stem} in {mask_dir}")

    mask_img = load_gray(mask_path)

    # target_size is PIL size: (W, H)
    if mask_img.size != target_size:
        mask_img = mask_img.resize(target_size, Image.Resampling.NEAREST)

    mask = np.asarray(mask_img).astype(np.float32) / 255.0
    return mask > 0.5, mask_path

def alpha_metrics(alpha, gt_mask):
    best = {
        "best_alpha_thr": np.nan,
        "best_alpha_iou": np.nan,
        "best_alpha_dice": np.nan,
    }

    best_dice = -1.0

    for thr in THRESHOLDS:
        pred_mask = alpha > thr
        inter = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        gt_area = gt_mask.sum()
        pred_area = pred_mask.sum()

        iou = inter / union if union > 0 else np.nan
        dice = (2 * inter) / (gt_area + pred_area) if (gt_area + pred_area) > 0 else np.nan

        if not np.isnan(dice) and dice > best_dice:
            best_dice = dice
            best = {
                "best_alpha_thr": thr,
                "best_alpha_iou": iou,
                "best_alpha_dice": dice,
            }

    return best

def one_pair(gt_path, pred_path, alpha_path, source):
    stem = gt_path.name.replace("_gt.png", "")

    gt_img_pil = Image.open(gt_path).convert("RGB")
    target_size = gt_img_pil.size

    gt = np.asarray(gt_img_pil).astype(np.float32) / 255.0
    pred = load_rgb(pred_path)

    if pred.shape != gt.shape:
        pred_pil = Image.open(pred_path).convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
        pred = np.asarray(pred_pil).astype(np.float32) / 255.0

    gt_mask, mask_path = find_mask(source, stem, target_size)

    full_mse = np.mean((gt - pred) ** 2)
    full_mae = np.mean(np.abs(gt - pred))
    full_psnr = psnr_from_mse(full_mse)

    if gt_mask.sum() > 0:
        diff = gt[gt_mask] - pred[gt_mask]
        plant_mse = np.mean(diff ** 2)
        plant_mae = np.mean(np.abs(diff))
        plant_psnr = psnr_from_mse(plant_mse)
    else:
        plant_mse = plant_mae = plant_psnr = np.nan

    box = bbox_from_mask(gt_mask, pad=20)
    if box is not None:
        y0, y1, x0, x1 = box
        gt_crop = gt[y0:y1, x0:x1]
        pred_crop = pred[y0:y1, x0:x1]

        bbox_mse = np.mean((gt_crop - pred_crop) ** 2)
        bbox_mae = np.mean(np.abs(gt_crop - pred_crop))
        bbox_psnr = psnr_from_mse(bbox_mse)

        if HAS_SKIMAGE and gt_crop.shape[0] >= 7 and gt_crop.shape[1] >= 7:
            bbox_ssim = ssim_fn(gt_crop, pred_crop, channel_axis=2, data_range=1.0)
        else:
            bbox_ssim = np.nan
    else:
        bbox_mse = bbox_mae = bbox_psnr = bbox_ssim = np.nan

    result = {
        "name": stem,
        "mask_path": str(mask_path),
        "gt_pixels": int(gt_mask.sum()),
        "gt_fraction": float(gt_mask.sum() / gt_mask.size),
        "full_psnr": full_psnr,
        "full_mae": full_mae,
        "plant_psnr": plant_psnr,
        "plant_mae": plant_mae,
        "bbox_psnr": bbox_psnr,
        "bbox_mae": bbox_mae,
        "bbox_ssim": bbox_ssim,
    }

    if alpha_path.exists():
        alpha_pil = Image.open(alpha_path).convert("L")
        if alpha_pil.size != target_size:
            alpha_pil = alpha_pil.resize(target_size, Image.Resampling.BILINEAR)
        alpha = np.asarray(alpha_pil).astype(np.float32) / 255.0
        result.update(alpha_metrics(alpha, gt_mask))
    else:
        result.update({
            "best_alpha_thr": np.nan,
            "best_alpha_iou": np.nan,
            "best_alpha_dice": np.nan,
        })

    return result

def summarize(rows, label):
    print(f"\n=== SUMMARY: {label} ===")
    keys = [
        "gt_fraction",
        "full_psnr", "full_mae",
        "plant_psnr", "plant_mae",
        "bbox_psnr", "bbox_mae", "bbox_ssim",
        "best_alpha_thr", "best_alpha_iou", "best_alpha_dice",
    ]
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            print(f"{k:18s}: mean={vals.mean():.4f} std={vals.std():.4f}")
        else:
            print(f"{k:18s}: nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--source", default="/cluster/project/cropsci/jmercoli/4dgs_project/data/4d_data")
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    eval_root = Path(args.eval_root)
    out_csv = eval_root / f"metrics_with_original_masks_{args.name}.csv"

    rows = []

    for split in ["test", "train"]:
        d = eval_root / split
        if not d.exists():
            continue

        for gt_path in sorted(d.glob("*_gt.png")):
            stem = gt_path.name.replace("_gt.png", "")
            pred_path = d / f"{stem}_pred.png"
            alpha_path = d / f"{stem}_alpha.png"

            if not pred_path.exists():
                print("Missing pred:", pred_path)
                continue

            r = one_pair(gt_path, pred_path, alpha_path, args.source)
            r["split"] = split
            rows.append(r)

    if not rows:
        raise RuntimeError(f"No gt/pred pairs found in {eval_root}")

    fieldnames = ["split"] + list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", out_csv)
    print("Rows:", len(rows))

    print("\n=== PER IMAGE ===")
    for r in rows:
        print(
            f"{r['split']:5s} {r['name'][:60]:60s} "
            f"plant_PSNR={r['plant_psnr']:.2f} "
            f"bbox_SSIM={r['bbox_ssim']:.3f} "
            f"best_IoU={r['best_alpha_iou']:.3f} "
            f"best_Dice={r['best_alpha_dice']:.3f} "
            f"thr={r['best_alpha_thr']:.2f}"
        )

    summarize(rows, args.name)

if __name__ == "__main__":
    main()
