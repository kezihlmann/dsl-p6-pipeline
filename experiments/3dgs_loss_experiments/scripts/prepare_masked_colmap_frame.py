from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def find_frame_root(extract_root: Path, frame_name: str) -> Path:
    candidates = [p for p in extract_root.rglob(frame_name) if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"Could not find folder named {frame_name} in extracted zip")
    return candidates[0]


def normalize_sparse(frame_root: Path) -> None:
    sparse_root = frame_root / "sparse"
    if not sparse_root.exists():
        raise FileNotFoundError(f"No sparse directory found in {frame_root}")

    zero_dir = sparse_root / "0"
    if zero_dir.exists():
        return

    zero_dir.mkdir(exist_ok=True)
    for path in list(sparse_root.iterdir()):
        if path.name != "0":
            shutil.move(str(path), str(zero_dir / path.name))


def preserve_masks(frame_root: Path) -> None:
    masks_dir = frame_root / "masks"
    gt_dir = frame_root / "masks_binary_gt"

    if masks_dir.exists() and not gt_dir.exists():
        shutil.move(str(masks_dir), str(gt_dir))
    elif not gt_dir.exists():
        raise FileNotFoundError(f"No masks or masks_binary_gt directory found in {frame_root}")

    created = 0
    for sam3_path in gt_dir.glob("*_mask_sam3.png"):
        gt_path = gt_dir / sam3_path.name.replace("_mask_sam3.png", "_mask_ground_truth.png")
        if not gt_path.exists():
            shutil.copy2(sam3_path, gt_path)
            created += 1

    active_dir = frame_root / "masks_binary_active"
    if active_dir.exists() or active_dir.is_symlink():
        if active_dir.is_symlink():
            active_dir.unlink()
        else:
            shutil.rmtree(active_dir)
    os.symlink(gt_dir, active_dir, target_is_directory=True)

    print(f"Active masks: {active_dir} -> {active_dir.resolve()}")
    print(f"Created {created} _mask_ground_truth.png copies")


def create_ones_masks(frame_root: Path) -> None:
    images_dir = frame_root / "images"
    ones_dir = frame_root / "masks_binary_ones"
    if not images_dir.exists():
        raise FileNotFoundError(f"No images directory found in {frame_root}")

    if ones_dir.exists():
        shutil.rmtree(ones_dir)
    ones_dir.mkdir(parents=True)

    count = 0
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        image = Image.open(image_path)
        ones = np.full((image.size[1], image.size[0]), 255, dtype=np.uint8)
        Image.fromarray(ones).save(ones_dir / f"{image_path.stem}_mask_ground_truth.png")
        count += 1

    print(f"Created {count} all-one masks in {ones_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one masked COLMAP frame for Wheat-3DGS.")
    parser.add_argument("--frame-zip", required=True, type=Path)
    parser.add_argument("--frame-name", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--tmp-root", type=Path, default=Path("/tmp/wheat_3dgs_frame_extract"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    frame_root = args.data_root / args.frame_name
    if frame_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{frame_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(frame_root)

    if args.tmp_root.exists():
        shutil.rmtree(args.tmp_root)
    args.tmp_root.mkdir(parents=True)
    args.data_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.frame_zip, "r") as zip_file:
        zip_file.extractall(args.tmp_root)

    extracted_frame = find_frame_root(args.tmp_root, args.frame_name)
    shutil.move(str(extracted_frame), str(frame_root))

    normalize_sparse(frame_root)
    preserve_masks(frame_root)
    create_ones_masks(frame_root)

    print("Prepared frame:", frame_root)


if __name__ == "__main__":
    main()
