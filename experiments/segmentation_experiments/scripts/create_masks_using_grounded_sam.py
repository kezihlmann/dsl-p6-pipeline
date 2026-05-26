from pathlib import Path
import os
import sys
import cv2
import numpy as np
import torch

# Make local GroundingDINO and segment_anything packages importable without a pip install.
_repo_root = Path(__file__).resolve().parent / "Grounded-Segment-Anything"
for _p in [
    str(_repo_root / "GroundingDINO"),       # provides: groundingdino
    str(_repo_root / "segment_anything"),    # provides: segment_anything (nested repo)
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor


# -----------------------
# Settings you may tweak
# -----------------------

# Root folder that contains timestep_* sub-directories with images and masks.
EVAL_MASK_DIR = Path("eval_mask")

# Weights / configs
GDINO_CONFIG = "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GDINO_CKPT = Path("weights/groundingdino_swint_ogc.pth")

# CPU-friendly SAM checkpoint is vit_b:
SAM_TYPE = "vit_l"
SAM_CKPT = Path("weights/sam_vit_l_0b3195.pth")  # change if you have a different SAM ckpt

# Text prompt for GroundingDINO
# TEXT_PROMPT = "plant, leaf, foliage"
TEXT_PROMPT = "green leaves"

# Thresholds (lower if it finds nothing; raise if too many false detections)
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20

# Save overlay previews (mask drawn over image)
SAVE_COMPRESSED_OVERLAYS = True


def ensure_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path.resolve()}")


def dino_boxes_to_xyxy_pixel(boxes_cxcywh_norm: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """
    GroundingDINO commonly returns boxes as normalized cx,cy,w,h in [0..1].
    Convert to pixel xyxy.
    """
    boxes = boxes_cxcywh_norm.clone()
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x0 = (cx - bw / 2.0) * w
    y0 = (cy - bh / 2.0) * h
    x1 = (cx + bw / 2.0) * w
    y1 = (cy + bh / 2.0) * h

    xyxy = torch.stack([x0, y0, x1, y1], dim=1)
    return xyxy


def pick_best_box(boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Pick the highest-score box (best single plant candidate)."""
    best_idx = int(torch.argmax(scores).item())
    return boxes[best_idx:best_idx + 1], scores[best_idx].item()


def save_mask_png(mask01: np.ndarray, out_path: Path):
    """
    mask01: bool or {0,1} array HxW
    Saves as 0..255 PNG (uncompressed for later processing)
    """
    m = (mask01.astype(np.uint8) * 255)
    cv2.imwrite(str(out_path), m)


def save_overlay(rgb: np.ndarray, mask01: np.ndarray, out_path: Path):
    """
    Simple overlay: green-ish mask blended onto original.
    Saves as heavily compressed JPEG (~10% of original size).
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = bgr.copy()

    # Put mask into green channel
    overlay[mask01, 1] = 255  # green channel

    blended = cv2.addWeighted(bgr, 0.70, overlay, 0.30, 0)
    # JPEG quality: 50 gives ~10% of original size (good for visual inspection only)
    cv2.imwrite(str(out_path), blended, [cv2.IMWRITE_JPEG_QUALITY, 50])


def main():
    device = "cpu"
    torch.set_grad_enabled(False)

    if not EVAL_MASK_DIR.exists():
        print(f"No eval_mask folder found: {EVAL_MASK_DIR.resolve()}")
        return

    ensure_exists(Path(GDINO_CONFIG), "GroundingDINO config")
    ensure_exists(GDINO_CKPT, "GroundingDINO checkpoint")
    ensure_exists(SAM_CKPT, "SAM checkpoint")

    # Collect all source images from every timestep_* sub-folder
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted(
        p
        for ts_dir in sorted(EVAL_MASK_DIR.glob("timestep_*"))
        for p in ts_dir.iterdir()
        if p.suffix.lower() in exts and not p.stem.endswith(("_mask_ground_truth",
                                                              "_mask_sam3",
                                                              "_mask_colors",
                                                              "_mask_grounded_sam_vit_b",
                                                              "_mask_grounded_sam_vit_l",
                                                              "_mask_grounded_sam_vit_h"))
    )

    if not images:
        print(f"No images found under: {EVAL_MASK_DIR.resolve()}")
        return

    print(f"Found {len(images)} images across timestep folders under: {EVAL_MASK_DIR.resolve()}")

    # Load models once
    print("Loading GroundingDINO (CPU)...")
    gdino = load_model(GDINO_CONFIG, str(GDINO_CKPT), device=device)

    print("Loading SAM (CPU)...")
    sam = sam_model_registry[SAM_TYPE](checkpoint=str(SAM_CKPT))
    sam.to(device)
    predictor = SamPredictor(sam)

    for img_path in images:
        print(f"\nProcessing: {img_path.name}")

        image_source, image_tensor = load_image(str(img_path))
        h, w = image_source.shape[:2]

        boxes, scores, phrases = predict(
            model=gdino,
            image=image_tensor,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device,
        )

        if boxes is None or len(boxes) == 0:
            print("  - No boxes found. Try lowering BOX_THRESHOLD/TEXT_THRESHOLD or adjusting TEXT_PROMPT.")
            continue

        best_box_norm, best_score = pick_best_box(boxes, scores)
        print(f"  - Best detection score: {best_score:.3f} | phrase: {phrases[int(torch.argmax(scores).item())]}")

        best_box_xyxy = dino_boxes_to_xyxy_pixel(best_box_norm, w=w, h=h)

        predictor.set_image(image_source)
        sam_box = predictor.transform.apply_boxes_torch(best_box_xyxy, image_source.shape[:2]).to(device)

        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=sam_box,
            multimask_output=False,
        )

        mask01 = masks[0, 0].detach().cpu().numpy().astype(bool)

        # Save mask alongside the source image in the same timestep folder
        out_mask = img_path.parent / f"{img_path.stem}_mask_grounded_sam_vit_b.png"
        save_mask_png(mask01, out_mask)
        print(f"  - Saved mask: {out_mask.name}")

        if SAVE_COMPRESSED_OVERLAYS:
            out_ov = img_path.parent / f"{img_path.stem}_mask_grounded_sam_vit_b_overlay_compressed.jpg"
            save_overlay(image_source, mask01, out_ov)
            print(f"  - Saved overlay: {out_ov.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()