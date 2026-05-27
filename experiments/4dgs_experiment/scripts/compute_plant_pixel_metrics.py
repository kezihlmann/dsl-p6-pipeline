from pathlib import Path
import numpy as np
from PIL import Image
import csv
import math

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32) / 255.0

def load_gray(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float32) / 255.0

def psnr_from_mse(mse):
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)

def bbox_from_mask(mask, pad=10):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    h, w = mask.shape
    y0 = max(0, y0 - pad)
    y1 = min(h - 1, y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(w - 1, x1 + pad)
    return y0, y1 + 1, x0, x1 + 1

def metrics_for_pair(gt_path, pred_path, alpha_path):
    gt = load_rgb(gt_path)
    pred = load_rgb(pred_path)
    alpha = load_gray(alpha_path)

    # GT image is already masked: background is black.
    # Threshold above pure black to define plant pixels.
    gt_mask = (gt.sum(axis=2) > (10 / 255.0))

    # Full-image metrics
    full_mse = np.mean((gt - pred) ** 2)
    full_mae = np.mean(np.abs(gt - pred))
    full_psnr = psnr_from_mse(full_mse)

    # Plant-pixel metrics
    if gt_mask.sum() > 0:
        diff_plant = gt[gt_mask] - pred[gt_mask]
        plant_mse = np.mean(diff_plant ** 2)
        plant_mae = np.mean(np.abs(diff_plant))
        plant_psnr = psnr_from_mse(plant_mse)
    else:
        plant_mse = plant_mae = plant_psnr = float("nan")

    # Bbox metrics: less harsh than pure plant pixels, but avoids huge background bias.
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
            bbox_ssim = float("nan")
    else:
        bbox_mse = bbox_mae = bbox_psnr = bbox_ssim = float("nan")

    # Alpha / silhouette metric
    # Use a low threshold because rendered alpha is soft.
    pred_mask = alpha > 0.10
    inter = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    gt_area = gt_mask.sum()
    pred_area = pred_mask.sum()

    alpha_iou = inter / union if union > 0 else float("nan")
    alpha_dice = (2 * inter) / (gt_area + pred_area) if (gt_area + pred_area) > 0 else float("nan")

    return {
        "name": gt_path.name.replace("_gt.png", ""),
        "gt_pixels": int(gt_area),
        "gt_fraction": float(gt_area / gt_mask.size),
        "pred_alpha_pixels": int(pred_area),
        "full_psnr": full_psnr,
        "full_mae": full_mae,
        "plant_psnr": plant_psnr,
        "plant_mae": plant_mae,
        "bbox_psnr": bbox_psnr,
        "bbox_mae": bbox_mae,
        "bbox_ssim": bbox_ssim,
        "alpha_iou": alpha_iou,
        "alpha_dice": alpha_dice,
    }

def summarize(rows):
    keys = [
        "gt_fraction",
        "full_psnr", "full_mae",
        "plant_psnr", "plant_mae",
        "bbox_psnr", "bbox_mae", "bbox_ssim",
        "alpha_iou", "alpha_dice",
    ]
    print("\n=== MEAN METRICS ===")
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals):
            print(f"{k:16s}: mean={vals.mean():.4f}  std={vals.std():.4f}")
        else:
            print(f"{k:16s}: nan")

def main():
    base = Path("/cluster/project/cropsci/jmercoli/4dgs_project/outputs/close10_dynamic_sam3mask_3000/eval_renders/iter_3000")
    out_csv = base / "plant_pixel_metrics_iter3000.csv"

    rows = []

    for split in ["test", "train"]:
        d = base / split
        for gt_path in sorted(d.glob("*_gt.png")):
            stem = gt_path.name.replace("_gt.png", "")
            pred_path = d / f"{stem}_pred.png"
            alpha_path = d / f"{stem}_alpha.png"

            if not pred_path.exists() or not alpha_path.exists():
                print("Missing pred/alpha for", gt_path)
                continue

            r = metrics_for_pair(gt_path, pred_path, alpha_path)
            r["split"] = split
            rows.append(r)

    if not rows:
        raise RuntimeError("No render pairs found.")

    fieldnames = ["split", "name"] + [k for k in rows[0].keys() if k not in ["split", "name"]]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", out_csv)
    print("Rows:", len(rows))

    print("\n=== PER IMAGE ===")
    for r in rows:
        print(
            f"{r['split']:5s} {r['name'][:55]:55s} "
            f"plant_PSNR={r['plant_psnr']:.2f} "
            f"bbox_PSNR={r['bbox_psnr']:.2f} "
            f"bbox_SSIM={r['bbox_ssim']:.3f} "
            f"alpha_IoU={r['alpha_iou']:.3f} "
            f"Dice={r['alpha_dice']:.3f}"
        )

    summarize(rows)

if __name__ == "__main__":
    main()
