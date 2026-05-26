from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np


BASE_DIR = Path("eval_mask")


def parse_polygon_points(points_str: str) -> np.ndarray:
    points = []
    for point_str in points_str.split(";"):
        x_str, y_str = point_str.split(",")
        points.append([float(x_str), float(y_str)])
    return np.array(points, dtype=np.int32)

def process_xml_file(xml_path: Path, output_dir: Path) -> int:
    print(f"\nProcessing XML: {xml_path.relative_to(BASE_DIR)}")

    xml_content = xml_path.read_text(encoding="utf-8")
    root = ET.fromstring(xml_content)

    images = root.findall(".//image")
    print(f"Found {len(images)} images with annotations")

    masks_created = 0
    for img_elem in images:
        img_name = img_elem.get("name")
        width = int(img_elem.get("width"))
        height = int(img_elem.get("height"))

        if img_name is None:
            print("  Skipping image entry without a name")
            continue

        polygons = img_elem.findall("polygon")
        if not polygons:
            print(f"  Skipping {img_name}: no polygons found")
            continue

        combined_mask = np.zeros((height, width), dtype=np.uint8)
        for poly in polygons:
            points_str = poly.get("points")
            if not points_str:
                continue
            points = parse_polygon_points(points_str)
            cv2.fillPoly(combined_mask, [points.reshape((-1, 1, 2))], 255)

        img_stem = Path(img_name).stem
        mask_path = output_dir / f"{img_stem}_mask_ground_truth.png"
        cv2.imwrite(str(mask_path), combined_mask)
        masks_created += 1
        print(f"  Saved mask: {mask_path.name}")

    return masks_created


def main() -> None:
    print("=== Ground Truth Mask Generator ===")
    print(f"Scanning: {BASE_DIR.resolve()}\n")

    xml_files = sorted(BASE_DIR.glob("timestep_*/annotations.xml"))
    if not xml_files:
        print("No annotation XML files found.")
        print("Expected: eval_mask/timestep_*/annotations.xml")
        return

    print(f"Found {len(xml_files)} annotation file(s).\n")

    total_masks_created = 0
    for xml_file in xml_files:
        output_dir = xml_file.parent
        print(f"Timestep: {xml_file.parent.name}")
        print(f"Output folder: {output_dir.relative_to(BASE_DIR)}")
        total_masks_created += process_xml_file(xml_file, output_dir)
        print()

    print(
        f"Done. Read {len(xml_files)} annotation files and created "
        f"{total_masks_created} ground truth masks directly in {len(xml_files)} timestep folders."
    )


if __name__ == "__main__":
    main()
