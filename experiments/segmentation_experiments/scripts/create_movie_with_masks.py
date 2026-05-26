from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import cv2
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "OpenCV is required to run this script. Install it with 'pip install opencv-python'."
    ) from exc


DEFAULT_IMAGE_DIR = (
    Path(__file__).resolve().parent
    / "images_and_masks"
    / "maize_4"
    / "images_sam_3_movie"
)
DEFAULT_MASK_DIR = (
    Path(__file__).resolve().parent
    / "images_and_masks"
    / "maize_4"
    / "masks_sam_3_movie"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "movies"
DEFAULT_OUTPUT_NAME = "GX010014_timelapse_with_masks.mp4"
DEFAULT_FPS = 24.0
MAX_OUTPUT_BYTES = 100 * 1024 * 1024
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
CODEC_CANDIDATES = ("mp4v",)
SCALE_CANDIDATES = (0.5, 0.4, 0.3, 0.2)
TIMESTEP_PATTERN = re.compile(r"_(\d+)$")


def print_progress(current: int, total: int, label: str) -> None:
    total = max(total, 1)
    width = 30
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r{label}: [{bar}] {current}/{total} ({ratio * 100:5.1f}%)", end="", flush=True)
    if current >= total:
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a stacked movie from maize_4 images and corresponding SAM3 masks. "
            "The original image is shown on top and the mask with timestep label is shown on the bottom."
        )
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory containing the timelapse images. Default: {DEFAULT_IMAGE_DIR}",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=DEFAULT_MASK_DIR,
        help=f"Directory containing the SAM3 masks. Default: {DEFAULT_MASK_DIR}",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Frames per second for the output movie. Default: {DEFAULT_FPS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Folder where the video should be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Output video filename. Default: {DEFAULT_OUTPUT_NAME}",
    )
    parser.add_argument(
        "--max-size-mb",
        type=float,
        default=100.0,
        help="Maximum preferred file size in MB before the script retries with more compression.",
    )
    parser.add_argument(
        "--codecs",
        nargs="+",
        default=list(CODEC_CANDIDATES),
        help=(
            "Ordered list of codecs to try. Default avoids broken OpenH264 setups by using mp4v only."
        ),
    )
    return parser.parse_args()


def iter_image_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_key(image_path: Path) -> str:
    return image_path.stem


def mask_key(mask_path: Path) -> str:
    stem = mask_path.stem
    suffix = "_mask_sam3"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def extract_timestep(stem: str) -> str:
    match = TIMESTEP_PATTERN.search(stem)
    return match.group(1) if match else stem


def collect_frame_pairs(image_dir: Path, mask_dir: Path) -> tuple[list[tuple[Path, Path, str]], list[str]]:
    image_paths = iter_image_files(image_dir)
    mask_paths = iter_image_files(mask_dir)
    masks_by_key = {mask_key(path): path for path in mask_paths}

    frame_pairs: list[tuple[Path, Path, str]] = []
    missing_masks: list[str] = []
    total_images = len(image_paths)

    for index, image_path in enumerate(image_paths, start=1):
        key = image_key(image_path)
        mask_path = masks_by_key.get(key)
        if mask_path is None:
            missing_masks.append(image_path.name)
            print_progress(index, total_images, "Matching masks")
            continue

        frame_pairs.append((image_path, mask_path, extract_timestep(key)))
        print_progress(index, total_images, "Matching masks")

    return frame_pairs, missing_masks


def round_down_to_even(value: int) -> int:
    return max(2, value - (value % 2))


def load_image(image_path: Path):
    frame = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return frame


def ensure_bgr(frame):
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def resize_to(frame, width: int, height: int):
    interpolation = cv2.INTER_AREA if frame.shape[1] >= width or frame.shape[0] >= height else cv2.INTER_LINEAR
    if frame.shape[1] != width or frame.shape[0] != height:
        return cv2.resize(frame, (width, height), interpolation=interpolation)
    return frame


def draw_timestep_label(mask_frame, timestep: str):
    labeled = mask_frame.copy()
    label = f"timestep {timestep}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, min(labeled.shape[1], labeled.shape[0]) / 900.0)
    thickness = max(2, int(round(font_scale * 2)))
    padding = max(10, int(round(font_scale * 12)))
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    top_left = (padding, padding)
    bottom_right = (
        padding * 2 + text_width,
        padding * 2 + text_height + baseline,
    )
    cv2.rectangle(labeled, top_left, bottom_right, (0, 0, 0), thickness=-1)
    cv2.putText(
        labeled,
        label,
        (padding, padding + text_height),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return labeled


def compose_frame(image_path: Path, mask_path: Path, timestep: str, pane_width: int, pane_height: int):
    image_frame = resize_to(ensure_bgr(load_image(image_path)), pane_width, pane_height)
    mask_frame = resize_to(ensure_bgr(load_image(mask_path)), pane_width, pane_height)
    mask_frame = draw_timestep_label(mask_frame, timestep)
    return cv2.vconcat([image_frame, mask_frame])


def total_source_bytes(frame_pairs: list[tuple[Path, Path, str]]) -> int:
    total = 0
    for image_path, mask_path, _ in frame_pairs:
        total += image_path.stat().st_size
        total += mask_path.stat().st_size
    return total


def build_video(
    frame_pairs: list[tuple[Path, Path, str]],
    output_path: Path,
    fps: float,
    codec: str,
    scale: float,
) -> int:
    first_image = ensure_bgr(load_image(frame_pairs[0][0]))
    pane_height = round_down_to_even(int(first_image.shape[0] * scale))
    pane_width = round_down_to_even(int(first_image.shape[1] * scale))
    output_height = round_down_to_even(pane_height * 2)
    output_width = pane_width
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a writer for codec '{codec}'.")

    try:
        total_frames = len(frame_pairs)
        for index, (image_path, mask_path, timestep) in enumerate(frame_pairs, start=1):
            writer.write(compose_frame(image_path, mask_path, timestep, pane_width, pane_height))
            print_progress(index, total_frames, f"Encoding {codec} @ {scale:.2f}x")
    finally:
        writer.release()

    if not output_path.exists():
        raise RuntimeError(f"Video writer did not create an output file: {output_path}")

    return output_path.stat().st_size


def choose_best_video(
    frame_pairs: list[tuple[Path, Path, str]],
    output_path: Path,
    fps: float,
    max_output_bytes: int,
    codecs: list[str],
) -> tuple[Path, str, float, int]:
    temp_dir = output_path.parent / ".movie_build_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    best_candidate: Path | None = None
    best_codec = ""
    best_scale = 1.0
    best_size: int | None = None

    try:
        for scale in SCALE_CANDIDATES:
            scale_best_path: Path | None = None
            scale_best_codec = ""
            scale_best_size: int | None = None

            for codec in codecs:
                temp_output = temp_dir / f"{output_path.stem}_{codec}_{scale:.2f}{output_path.suffix}"
                if temp_output.exists():
                    temp_output.unlink()

                try:
                    size_bytes = build_video(frame_pairs, temp_output, fps, codec, scale)
                except RuntimeError as exc:
                    print(f"Skipping codec {codec} at scale {scale:.2f}: {exc}")
                    continue

                if scale_best_size is None or size_bytes < scale_best_size:
                    if scale_best_path is not None and scale_best_path.exists():
                        scale_best_path.unlink()
                    scale_best_path = temp_output
                    scale_best_codec = codec
                    scale_best_size = size_bytes
                elif temp_output.exists():
                    temp_output.unlink()

            if scale_best_path is None or scale_best_size is None:
                continue

            if best_size is None or scale_best_size < best_size:
                if best_candidate is not None and best_candidate.exists():
                    best_candidate.unlink()
                best_candidate = scale_best_path
                best_codec = scale_best_codec
                best_scale = scale
                best_size = scale_best_size
            elif scale_best_path.exists():
                scale_best_path.unlink()

            if best_size is not None and best_size <= max_output_bytes:
                break

        if best_candidate is None or best_size is None:
            raise RuntimeError("Failed to build the movie with any available codec.")

        if output_path.exists():
            output_path.unlink()
        best_candidate.replace(output_path)
        return output_path, best_codec, best_scale, best_size
    finally:
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except OSError:
                pass


def main() -> int:
    args = parse_args()
    max_output_bytes = int(args.max_size_mb * 1024 * 1024)
    output_path = args.output_dir / args.output_name

    frame_pairs, missing_masks = collect_frame_pairs(args.image_dir, args.mask_dir)
    if not frame_pairs:
        print(
            f"No matching image/mask pairs were found in {args.image_dir} and {args.mask_dir}.",
            file=sys.stderr,
        )
        return 1

    print(f"Collected {len(frame_pairs)} image/mask pairs.")
    if missing_masks:
        print(f"Skipped {len(missing_masks)} images because the matching mask was missing.")

    source_bytes = total_source_bytes(frame_pairs)

    output_path, codec, scale, size_bytes = choose_best_video(
        frame_pairs=frame_pairs,
        output_path=output_path,
        fps=args.fps,
        max_output_bytes=max_output_bytes,
        codecs=args.codecs,
    )

    print(f"Saved movie to: {output_path}")
    print(f"Codec used: {codec}")
    print(f"FPS: {args.fps}")
    print(f"Scale used for final movie: {scale:.2f}")
    print(f"File size: {size_bytes / (1024 * 1024):.2f} MB")
    print(f"Source size: {source_bytes / (1024 * 1024):.2f} MB")
    if size_bytes > 0:
        compression_ratio = source_bytes / size_bytes
        compression_percent = 100.0 * (1.0 - (size_bytes / source_bytes)) if source_bytes > 0 else 0.0
        print(f"Compression ratio: {compression_ratio:.2f}:1")
        print(f"Compression rate: {compression_percent:.2f}% smaller than the source files")
    if size_bytes > max_output_bytes:
        print(
            f"Warning: final file is larger than {args.max_size_mb:.1f} MB even after retrying with compression."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
