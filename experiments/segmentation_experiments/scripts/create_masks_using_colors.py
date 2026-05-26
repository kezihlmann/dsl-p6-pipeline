from pathlib import Path
import os
import cv2
import numpy as np


# -----------------------
# Settings you may tweak
# -----------------------

# Root folder that contains timestep_* sub-directories with images and masks.
EVAL_MASK_DIR = Path("eval_mask")

# HSV color range for green plants (you can adjust these)
# Lower bound for green
LOWER_GREEN = np.array([25, 30, 30])
# Upper bound for green
UPPER_GREEN = np.array([90, 255, 255])

# Morphological operations kernel size
KERNEL_SIZE = 5
ITERATIONS = 2

# JPEG compression quality for overlays
OVERLAY_QUALITY = 75


def create_basic_mask(image_path):
    """
    Create mask using basic image processing techniques.
    Uses HSV color space to detect green vegetation.
    
    Args:
        image_path: Path to input image
        
    Returns:
        mask: Binary mask of detected vegetation
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  ❌ Failed to read image: {image_path}")
        return None
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create mask for green colors
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
    
    # Close small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=ITERATIONS)
    
    # Open to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=ITERATIONS // 2)
    
    # Dilate to connect nearby components
    mask = cv2.dilate(mask, kernel, iterations=ITERATIONS // 2)
    
    return mask, image


def save_mask(mask, output_path):
    """Save mask as PNG image."""
    cv2.imwrite(str(output_path), mask)


def create_overlay(image, mask, output_path, quality=OVERLAY_QUALITY):
    """
    Create and save overlay image showing mask on original image.
    
    Args:
        image: Original RGB image
        mask: Binary mask
        output_path: Path to save overlay
        quality: JPEG compression quality
    """
    # Create colored overlay
    overlay = image.copy()
    
    # Convert mask to 3-channel
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Apply green color to masked regions
    overlay[mask > 0] = [0, 255, 0]  # Green in BGR
    
    # Blend original image with overlay
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Save as compressed JPEG
    cv2.imwrite(str(output_path), blended, 
                [cv2.IMWRITE_JPEG_QUALITY, quality])


def main():
    """Main processing function."""

    if not EVAL_MASK_DIR.exists():
        print(f"❌ eval_mask folder not found: {EVAL_MASK_DIR.resolve()}")
        return

    # Collect all images from every timestep_* sub-folder
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_files = sorted(
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

    if not image_files:
        print(f"❌ No images found under: {EVAL_MASK_DIR}")
        return

    print(f"Found {len(image_files)} images across timestep folders under: {EVAL_MASK_DIR}")

    processed = 0

    for image_path in image_files:
        image_name = image_path.stem
        print(f"Processing: {image_path.name}")

        result = create_basic_mask(image_path)
        if result is None:
            continue

        mask, image = result

        mask_pixels = np.sum(mask > 0)
        percentage = (mask_pixels / (mask.shape[0] * mask.shape[1])) * 100

        # Save mask alongside the source image in the same timestep folder
        mask_output_path = image_path.parent / f"{image_name}_mask_colors.png"
        save_mask(mask, mask_output_path)

        print(f"  - Detected vegetation: {percentage:.1f}% of image")
        print(f"  - Saved mask: {mask_output_path.name}")

        processed += 1

    print(f"\n✅ Successfully processed {processed}/{len(image_files)} images")


if __name__ == "__main__":
    main()
