from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch

try:
    import lpips
    HAS_LPIPS = True
except Exception:
    HAS_LPIPS = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB")).astype(np.float32) / 255.0


def load_mask(path, target_hw):
    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 3 and arr.shape[2] == 4:
        mask = arr[:, :, 3] > 127
    else:
        mask = np.array(img.convert("L")) > 127

    h, w = target_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def psnr(a, b, mask=None):
    if mask is not None:
        if mask.sum() == 0:
            return np.nan
        diff = a[mask] - b[mask]
    else:
        diff = a - b
    mse = np.mean(diff ** 2)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def mae(a, b, mask=None):
    if mask is not None:
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(np.abs(a[mask] - b[mask])))
    return float(np.mean(np.abs(a - b)))


def rgb_to_silhouette(rgb):
    # Rendered 3DGS plant is on black background, so non-black silhouette is appropriate.
    gray_energy = np.max(rgb, axis=2)
    sil = gray_energy > 0.03

    # Clean tiny isolated dots.
    sil_u8 = sil.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    sil_u8 = cv2.morphologyEx(sil_u8, cv2.MORPH_OPEN, kernel)
    sil_u8 = cv2.morphologyEx(sil_u8, cv2.MORPH_CLOSE, kernel)
    return sil_u8.astype(bool)


def iou_dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    iou = inter / union if union > 0 else np.nan
    dice = (2 * inter) / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else np.nan
    return float(iou), float(dice)


def lpips_score(model, pred, gt, device):
    # pred/gt HWC [0,1] -> NCHW [-1,1]
    p = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
    g = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2 - 1
    with torch.no_grad():
        return float(model(p, g).item())


def find_render_dirs(out):
    candidates = []
    for d in out.rglob("renders"):
        if d.is_dir() and "test" in str(d):
            candidates.append(d)
    return sorted(candidates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--iteration", default="15000")
    ap.add_argument("--output-csv", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    scene = Path(args.scene)

    render_dirs = find_render_dirs(out)
    if not render_dirs:
        raise RuntimeError(f"No test render dirs found under {out}")

    # Prefer directory containing the requested iteration.
    render_dir = None
    for d in render_dirs:
        if f"ours_{args.iteration}" in str(d) or f"iteration_{args.iteration}" in str(d):
            render_dir = d
            break
    if render_dir is None:
        render_dir = render_dirs[-1]

    base = render_dir.parent
    gt_dir = base / "gt"
    if not gt_dir.exists():
        # Sometimes GT is named ground_truth
        gt_dir = base / "ground_truth"

    if not gt_dir.exists():
        raise RuntimeError(f"Could not find GT dir near {render_dir}. Nearby dirs: {list(base.iterdir())}")

    print("Using render_dir:", render_dir)
    print("Using gt_dir:", gt_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = None
    if HAS_LPIPS:
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    else:
        print("[WARN] lpips package not available, LPIPS will be NaN")

    rows = []

    render_files = sorted(list(render_dir.glob("*.png")) + list(render_dir.glob("*.jpg")))
    if not render_files:
        raise RuntimeError(f"No render images in {render_dir}")

    for rp in render_files:
        stem = rp.stem

        # Match GT by same filename/stem.
        gt_candidates = list(gt_dir.glob(stem + ".*"))
        if not gt_candidates:
            # fallback: same index order
            idx = render_files.index(rp)
            gt_files = sorted(list(gt_dir.glob("*.png")) + list(gt_dir.glob("*.jpg")))
            if idx < len(gt_files):
                gt_path = gt_files[idx]
            else:
                print("[WARN] no GT for", rp)
                continue
        else:
            gt_path = gt_candidates[0]

        pred = load_rgb(rp)
        gt = load_rgb(gt_path)

        if pred.shape != gt.shape:
            gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_AREA)

        h, w = pred.shape[:2]

        # Try to find SAM/mask by camera image name.
        # GT dir images are usually named like camera files, so use gt stem.
        mask_candidates = []
        masks_dir = scene / "masks"
        if masks_dir.exists():
            mask_candidates += list(masks_dir.glob(gt_path.stem + ".*"))
            mask_candidates += list(masks_dir.glob(gt_path.stem + "_mask_sam3.*"))
            mask_candidates += list(masks_dir.glob(gt_path.stem.replace("_rgba", "") + "_mask_sam3.*"))

        if mask_candidates:
            mask = load_mask(mask_candidates[0], (h, w))
        else:
            # Fallback: if GT is already black-background masked, infer mask from GT nonblack.
            mask = np.max(gt, axis=2) > 0.03
            print("[WARN] using GT nonblack as mask for", gt_path.name)

        pred_sil = rgb_to_silhouette(pred)
        iou, dice = iou_dice(pred_sil, mask)

        row = {
            "image": rp.name,
            "render_path": str(rp),
            "gt_path": str(gt_path),
            "mask_fg_fraction": float(mask.mean()),
            "pred_fg_fraction": float(pred_sil.mean()),
            "full_psnr": psnr(pred, gt),
            "plant_psnr": psnr(pred, gt, mask),
            "full_mae": mae(pred, gt),
            "plant_mae": mae(pred, gt, mask),
            "silhouette_iou": iou,
            "silhouette_dice": dice,
        }

        if HAS_SKIMAGE:
            row["full_ssim"] = float(ssim_fn(gt, pred, channel_axis=2, data_range=1.0))
        else:
            row["full_ssim"] = np.nan

        if lpips_model is not None:
            row["lpips"] = lpips_score(lpips_model, pred, gt, device)
        else:
            row["lpips"] = np.nan

        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = df.select_dtypes(include=[np.number]).mean().to_dict()
    summary_std = df.select_dtypes(include=[np.number]).std().to_dict()

    print("\nSaved:", out_csv)
    print("\nMean:")
    for k, v in summary.items():
        print(f"  {k}: {v:.6f}")

    print("\nStd:")
    for k, v in summary_std.items():
        print(f"  {k}: {v:.6f}")

    with open(out_csv.with_suffix(".summary.json"), "w") as f:
        json.dump({"mean": summary, "std": summary_std}, f, indent=2)


if __name__ == "__main__":
    main()
