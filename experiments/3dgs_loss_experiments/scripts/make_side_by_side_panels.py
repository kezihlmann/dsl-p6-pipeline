from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_mask_rgb(path: Path) -> Image.Image:
    mask = np.array(Image.open(path).convert("L")) > 127
    return Image.fromarray((mask.astype(np.uint8) * 255)).convert("RGB")


def render_silhouette(path: Path, threshold: int) -> Image.Image:
    rgb = np.array(Image.open(path).convert("RGB"))
    sil = rgb.max(axis=2) > threshold
    return Image.fromarray((sil.astype(np.uint8) * 255)).convert("RGB")


def resize_same_height(image: Image.Image, target_h: int) -> Image.Image:
    width, height = image.size
    return image.resize((int(round(width * target_h / height)), target_h))


def add_title(image: Image.Image, title: str, title_h: int = 40) -> Image.Image:
    canvas = Image.new("RGB", (image.size[0], image.size[1] + title_h), "white")
    canvas.paste(image, (0, title_h))
    ImageDraw.Draw(canvas).text((10, 10), title, fill="black")
    return canvas


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png"):
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    for candidate in [
        mask_dir / f"{stem}_mask_sam3.png",
        mask_dir / f"{stem}_mask_ground_truth.png",
        mask_dir / f"{stem}_sam3.png",
        mask_dir / f"{stem}_mask.png",
    ]:
        if candidate.exists():
            return candidate
    matches = sorted(mask_dir.glob(f"{stem}*.png"))
    return matches[0] if matches else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create side-by-side panels for a 3DGS loss experiment.")
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--iteration", type=int, default=15000)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--silhouette-threshold", type=int, default=10)
    args = parser.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    images_dir = args.scene / "images"
    gt_mask_dir = args.scene / "masks_binary_gt"
    sam3_dir = args.scene / "masks_sam3"
    if not sam3_dir.exists():
        sam3_dir = gt_mask_dir
    render_dir = args.model / "test" / f"ours_{args.iteration}" / "renders"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for render_name, stem in mapping.items():
        image_path = find_image(images_dir, stem)
        gt_mask_path = gt_mask_dir / f"{stem}_mask_ground_truth.png"
        sam3_path = find_mask(sam3_dir, stem)
        render_path = render_dir / render_name
        if image_path is None or not gt_mask_path.exists() or sam3_path is None or not render_path.exists():
            continue

        panels = [
            add_title(resize_same_height(load_rgb(image_path), args.height), "GT RGB"),
            add_title(resize_same_height(load_mask_rgb(gt_mask_path), args.height), "GT silhouette"),
            add_title(resize_same_height(load_rgb(render_path), args.height), "3DGS render"),
            add_title(resize_same_height(render_silhouette(render_path, args.silhouette_threshold), args.height), "3DGS silhouette"),
            add_title(resize_same_height(load_mask_rgb(sam3_path), args.height), "SAM3 mask"),
        ]
        gap = 12
        width = sum(panel.size[0] for panel in panels) + gap * (len(panels) - 1)
        height = max(panel.size[1] for panel in panels)
        combined = Image.new("RGB", (width, height), "white")
        x = 0
        for panel in panels:
            combined.paste(panel, (x, 0))
            x += panel.size[0] + gap
        combined.save(args.out_dir / f"{Path(render_name).stem}_comparison.png")
        count += 1

    print(f"Wrote {count} panels to {args.out_dir}")


if __name__ == "__main__":
    main()
