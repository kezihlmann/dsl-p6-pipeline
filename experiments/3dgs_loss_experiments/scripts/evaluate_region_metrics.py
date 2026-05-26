from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None

try:
    import lpips
except Exception:
    lpips = None

try:
    from skimage.metrics import structural_similarity
except Exception:
    structural_similarity = None


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def load_mask(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L")) > 127


def psnr(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = gt.astype(np.float32) - pred.astype(np.float32)
    if mask is not None:
        diff = diff[np.repeat(mask[:, :, None], 3, axis=2)]
    if diff.size == 0:
        return float("nan")
    mse = float(np.mean(diff ** 2))
    if mse <= 1e-12:
        return 99.0
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def mae(gt: np.ndarray, pred: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(gt.astype(np.float32) - pred.astype(np.float32))
    if mask is not None:
        diff = diff[np.repeat(mask[:, :, None], 3, axis=2)]
    if diff.size == 0:
        return float("nan")
    return float(np.mean(diff))


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def crop(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return arr[y0:y1, x0:x1, ...] if arr.ndim == 3 else arr[y0:y1, x0:x1]


def ssim_value(gt: np.ndarray, pred: np.ndarray) -> float:
    if structural_similarity is None:
        return float("nan")
    return float(structural_similarity(gt, pred, channel_axis=2, data_range=255))


def make_lpips_model(device: str):
    if lpips is None or torch is None:
        return None
    return lpips.LPIPS(net="vgg").to(device).eval()


def lpips_value(model, gt: np.ndarray, pred: np.ndarray, device: str, size: int = 512) -> float:
    if model is None:
        return float("nan")
    gt_img = Image.fromarray(gt).resize((size, size))
    pred_img = Image.fromarray(pred).resize((size, size))
    gt_t = torch.tensor(np.array(gt_img)).permute(2, 0, 1).float() / 127.5 - 1
    pred_t = torch.tensor(np.array(pred_img)).permute(2, 0, 1).float() / 127.5 - 1
    with torch.no_grad():
        return float(model(pred_t.unsqueeze(0).to(device), gt_t.unsqueeze(0).to(device)).item())


def segmentation_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"),
        "dice": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
    }


def infer_test_names(repo: Path, scene_path: Path, resolution: int, model_path: Path) -> list[str]:
    sys.path.insert(0, str(repo))
    from argparse import ArgumentParser

    from arguments import ModelParams
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    parser = ArgumentParser()
    ModelParams(parser, sentinel=True)
    args = parser.parse_args([])
    args.source_path = str(scene_path)
    args.model_path = str(model_path)
    args.images = "images"
    args.eval = True
    args.resolution = resolution
    args.data_device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    args.white_background = False

    scene = Scene(args, GaussianModel(3), load_iteration=None, shuffle=False)
    return [camera.image_name for camera in scene.getTestCameras()]


def summarize(rows: list[dict[str, object]], keys: list[str]) -> dict[str, float]:
    summary = {}
    for key in keys:
        values = []
        for row in rows:
            value = float(row[key])
            if not math.isnan(value):
                values.append(value)
        summary[f"{key}_mean"] = float(np.mean(values)) if values else float("nan")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full, plant, bbox, and silhouette metrics for a 3DGS output.")
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=15000)
    parser.add_argument("--resolution", type=int, default=4)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--silhouette-threshold", type=int, default=10)
    args = parser.parse_args()

    render_dir = args.model / "test" / f"ours_{args.iteration}" / "renders"
    gt_dir = args.model / "test" / f"ours_{args.iteration}" / "gt"
    mask_dir = args.mask_dir or args.scene / "masks_binary_gt"
    if not render_dir.exists():
        raise FileNotFoundError(render_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)
    if not mask_dir.exists():
        raise FileNotFoundError(mask_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    test_names = infer_test_names(args.repo, args.scene, args.resolution, args.model)
    index_to_stem = {f"{idx:05d}.png": stem for idx, stem in enumerate(test_names)}

    device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
    lp_model = make_lpips_model(device)

    rows = []
    for render_path in sorted(render_dir.glob("*.png")):
        gt_path = gt_dir / render_path.name
        stem = index_to_stem.get(render_path.name)
        if stem is None or not gt_path.exists():
            continue
        mask_path = mask_dir / f"{stem}_mask_ground_truth.png"
        if not mask_path.exists():
            continue

        pred = load_rgb(render_path)
        gt = load_rgb(gt_path)
        mask = load_mask(mask_path)
        if mask.shape != pred.shape[:2]:
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
            mask = np.array(mask_img.resize((pred.shape[1], pred.shape[0]), Image.NEAREST)) > 127

        bbox = bbox_from_mask(mask)
        if bbox is None:
            continue

        pred_sil = pred.max(axis=2) > args.silhouette_threshold
        gt_masked = gt.copy()
        pred_masked = pred.copy()
        gt_masked[~mask] = 0
        pred_masked[~mask] = 0
        gt_bbox = crop(gt, bbox)
        pred_bbox = crop(pred, bbox)

        rows.append({
            "file": render_path.name,
            "image_stem": stem,
            "mask_foreground_pixels": int(mask.sum()),
            "bbox_x0": bbox[0],
            "bbox_y0": bbox[1],
            "bbox_x1": bbox[2],
            "bbox_y1": bbox[3],
            "full_psnr": psnr(gt, pred),
            "full_mae": mae(gt, pred),
            "full_ssim": ssim_value(gt, pred),
            "full_lpips": lpips_value(lp_model, gt, pred, device),
            "plant_psnr": psnr(gt, pred, mask),
            "plant_mae": mae(gt, pred, mask),
            "plant_ssim": ssim_value(gt_masked, pred_masked),
            "plant_lpips": lpips_value(lp_model, gt_masked, pred_masked, device),
            "bbox_psnr": psnr(gt_bbox, pred_bbox),
            "bbox_mae": mae(gt_bbox, pred_bbox),
            "bbox_ssim": ssim_value(gt_bbox, pred_bbox),
            "bbox_lpips": lpips_value(lp_model, gt_bbox, pred_bbox, device),
            **segmentation_metrics(pred_sil, mask),
        })

    if not rows:
        raise RuntimeError("No images evaluated. Check paths and mask naming.")

    metric_keys = [
        "full_psnr", "full_mae", "full_ssim", "full_lpips",
        "plant_psnr", "plant_mae", "plant_ssim", "plant_lpips",
        "bbox_psnr", "bbox_mae", "bbox_ssim", "bbox_lpips",
        "iou", "dice", "precision", "recall",
    ]

    with (args.out_dir / "per_image_region_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (args.out_dir / "summary_region_metrics.json").write_text(json.dumps(summarize(rows, metric_keys), indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "test_camera_mapping.json").write_text(json.dumps(index_to_stem, indent=2) + "\n", encoding="utf-8")
    print("Wrote metrics to:", args.out_dir)


if __name__ == "__main__":
    main()
