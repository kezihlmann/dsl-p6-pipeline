from pathlib import Path
import argparse
import re
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg


BASE_DIR = Path("images_and_masks/maize_2")
FALLBACK_IMAGE_BASE_DIR = Path("initial_data_sample/maize_2")


TOP_ROW_SPECS = [
    ("Original", None),
    ("Ground Truth", "_mask_ground_truth.png"),
]

BOTTOM_ROW_SPECS = [
    ("Color index", "_mask_colors.png", "colors"),
    ("Grounded SAM basic", "_mask_grounded_sam_vit_b.png", "grounded_sam_vit_b"),
    ("Grounded SAM large", "_mask_grounded_sam_vit_l.png", "grounded_sam_vit_l"),
    ("SAM 3", "_mask_sam3.png", "sam3"),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a 6-panel comparison plot for one frame/image index: "
            "original, GT, colors, GroundedSAM vit_b, GroundedSAM vit_l, SAM3."
        )
    )
    parser.add_argument("--frame-id", default="frame_1180", help="Frame folder id, e.g. frame_1180")
    parser.add_argument("--frame-number", default="1880", help="Frame number inside filenames, e.g. 1880")
    parser.add_argument("--image-index", type=int, default=6, help="Image index inside filenames, e.g. 6")
    parser.add_argument(
        "--bbox-pad",
        type=int,
        default=20,
        help="Padding (pixels) added around SAM3 mask bounding box.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Default: images_and_masks/maize_2/<frame-id>_plots/<stem>_comparison_6panel.png",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Optional path to mask quality report. Default: mask_quality_report_<frame-id>.txt",
    )
    return parser.parse_args()


def find_stem(frame_masks_dir: Path, frame_number: str, image_index: int) -> str:
    candidates = sorted(frame_masks_dir.glob(f"*_mask_ground_truth.png"))
    suffix = f"_{frame_number}_{image_index}_mask_ground_truth"

    for p in candidates:
        if p.stem.endswith(suffix):
            return p.stem.replace("_mask_ground_truth", "")

    raise FileNotFoundError(
        f"Could not find ground-truth mask matching frame_number={frame_number}, image_index={image_index} in {frame_masks_dir}"
    )


def load_rgb_image(image_path: Path):
    try:
        image = mpimg.imread(str(image_path))
    except Exception as exc:
        raise FileNotFoundError(f"Could not read image: {image_path}") from exc

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Keep only RGB channels if alpha is present.
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    return image


def load_gray_mask(mask_path: Path):
    try:
        mask = mpimg.imread(str(mask_path))
    except Exception as exc:
        raise FileNotFoundError(f"Could not read mask: {mask_path}") from exc

    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    # Convert RGB/RGBA masks to single channel for consistent display.
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    # Normalize to 0..255 for consistent grayscale plotting across file types.
    if mask.max() <= 1.0:
        mask = mask * 255.0
    return mask


def sam3_bbox_from_mask(mask: "np.ndarray", pad: int = 0):
    ys, xs = (mask > 127).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("SAM3 mask has no foreground pixels; cannot compute crop box.")

    x_min = max(int(xs.min()) - pad, 0)
    x_max = min(int(xs.max()) + pad + 1, mask.shape[1])
    y_min = max(int(ys.min()) - pad, 0)
    y_max = min(int(ys.max()) + pad + 1, mask.shape[0])
    return x_min, y_min, x_max, y_max


def crop_to_bbox(arr, bbox):
    x_min, y_min, x_max, y_max = bbox
    if arr.ndim == 2:
        return arr[y_min:y_max, x_min:x_max]
    return arr[y_min:y_max, x_min:x_max, :]


def load_metrics_from_report(report_path: Path, image_id: str):
    if not report_path.exists():
        return {}

    # Example row start:
    # GX..._6 grounded_sam_vit_b GroundedSAM vit_b 0.856117 ...
    row_pattern = re.compile(r"^\s*(\S+)\s+(\S+)\s+(.+?)\s+([0-9]*\.[0-9]+)\s+.*?([0-9]*\.[0-9]+)\s*$")

    metrics_by_method = {}
    in_table = False
    with report_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "[PER-IMAGE RESULTS]" in line:
                in_table = True
                continue
            if not in_table:
                continue

            m = row_pattern.match(line)
            if not m:
                continue

            row_image_id = m.group(1)
            method_key = m.group(2)
            iou = float(m.group(4))
            map_value = float(m.group(5))

            if row_image_id == image_id:
                metrics_by_method[method_key] = {
                    "iou": iou,
                    "map": map_value,
                }

    return metrics_by_method


def resolve_original_image(frame_id: str, stem: str) -> Path:
    in_masks_tree = BASE_DIR / frame_id / "images" / f"{stem}.jpg"
    if in_masks_tree.exists():
        return in_masks_tree

    in_fallback_tree = FALLBACK_IMAGE_BASE_DIR / frame_id / "images" / f"{stem}.jpg"
    if in_fallback_tree.exists():
        return in_fallback_tree

    # Try PNG fallback in both trees.
    png1 = BASE_DIR / frame_id / "images" / f"{stem}.png"
    png2 = FALLBACK_IMAGE_BASE_DIR / frame_id / "images" / f"{stem}.png"
    if png1.exists():
        return png1
    if png2.exists():
        return png2

    raise FileNotFoundError(
        "Original image not found in either "
        f"{in_masks_tree.parent} or {in_fallback_tree.parent} for stem {stem}."
    )


def main():
    args = parse_args()

    frame_masks_dir = BASE_DIR / f"{args.frame_id}_masks"
    if not frame_masks_dir.exists():
        raise FileNotFoundError(f"Masks folder not found: {frame_masks_dir}")

    stem = find_stem(frame_masks_dir, str(args.frame_number), args.image_index)
    original_path = resolve_original_image(args.frame_id, stem)
    report_path = Path(args.report_path) if args.report_path else Path(f"mask_quality_report_{args.frame_id}.txt")
    metric_values = load_metrics_from_report(report_path, stem)

    output_path = Path(args.output) if args.output else (
        BASE_DIR / f"{args.frame_id}_plots" / f"{stem}_comparison_6panel.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sam3_path = frame_masks_dir / f"{stem}_mask_sam3.png"
    sam3_mask = load_gray_mask(sam3_path)
    bbox = sam3_bbox_from_mask(sam3_mask, pad=max(args.bbox_pad, 0))

    fig = plt.figure(figsize=(13, 9), dpi=140)
    gs = GridSpec(2, 4, figure=fig, wspace=0.0, hspace=0.08)

    # Top row centered without nested grids: each panel spans two columns.
    top_axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4])]

    for ax, (title, suffix) in zip(top_axes, TOP_ROW_SPECS):
        if suffix is None:
            img = load_rgb_image(original_path)
            img = crop_to_bbox(img, bbox)
            ax.imshow(img)
            ax.set_title(title, fontsize=12)
        else:
            mask_path = frame_masks_dir / f"{stem}{suffix}"
            mask = load_gray_mask(mask_path)
            mask = crop_to_bbox(mask, bbox)
            ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
            ax.text(
                0.02,
                0.98,
                title,
                transform=ax.transAxes,
                color="white",
                fontsize=11,
                fontweight="bold",
                ha="left",
                va="top",
            )
        ax.axis("off")

    # Bottom row: full width with 4 panels.
    bottom_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    for ax, (title, suffix, method_key) in zip(bottom_axes, BOTTOM_ROW_SPECS):
        mask_path = frame_masks_dir / f"{stem}{suffix}"
        mask = load_gray_mask(mask_path)
        mask = crop_to_bbox(mask, bbox)
        ax.imshow(mask, cmap="gray", vmin=0, vmax=255)
        ax.text(
            0.02,
            0.98,
            title,
            transform=ax.transAxes,
            color="white",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="top",
        )

        metrics = metric_values.get(method_key, {})
        iou = metrics.get("iou")
        map_value = metrics.get("map")
        iou_text = f"IoU: {iou:.2f}" if iou is not None else "IoU: N/A"
        map_text = f"mAP: {map_value:.2f}" if map_value is not None else "mAP: N/A"
        ax.text(
            0.02,
            0.04,
            f"{iou_text}\n{map_text}",
            transform=ax.transAxes,
            color="white",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        ax.axis("off")

    fig.suptitle(
        f"Comparison for {stem} (frame_number={args.frame_number}, image_index={args.image_index})",
        fontsize=14,
    )
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.02, wspace=0.0, hspace=0.08)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved plot: {output_path}")


if __name__ == "__main__":
    main()
