# Segmentation Experiments

This folder contains the scripts used to generate, compare, and visualize plant masks for the segmentation part of the project.

The workflow is centered around a small evaluation dataset in `eval_mask/`, where each `timestep_*` folder contains source images, annotation XMLs, and the predicted masks written by the different methods.

## Contents

- `scripts/create_ground_truth_masks.py`: converts CVAT-style polygon annotations in `annotations.xml` into binary `*_mask_ground_truth.png` masks.
- `scripts/create_masks_using_colors.py`: creates simple HSV-based vegetation masks and writes `*_mask_colors.png`.
- `scripts/create_masks_using_grounded_sam.py`: runs GroundingDINO + SAM and writes `*_mask_grounded_sam_vit_b.png` plus optional overlay JPEGs.
- `scripts/create_masks_using_sam3.ipynb`: notebook used for the SAM3-based mask generation experiments.
- `scripts/analysis/analyze_mask_quality.py`: evaluates available predictions against the ground-truth masks and writes a text report with IoU, Dice, precision, recall, F1, and mAP.
- `scripts/analysis/create_plot.py`: makes a 6-panel comparison figure for one image.
- `scripts/create_movie_with_masks.py`: creates a stacked timelapse movie that shows each original image together with its mask.

## Data Layout

The evaluation scripts expect a structure like:

```text
eval_mask/
  timestep_0000/
    annotations.xml
    GX....jpg
    GX...._mask_ground_truth.png
    GX...._mask_colors.png
    GX...._mask_grounded_sam_vit_b.png
    GX...._mask_sam3.png
  timestep_0060/
    ...
```

Mask files are written next to the source images inside each timestep folder.

## Generate Ground Truth Masks

Run from the repository root:

```bash
python experiments/segmentation_experiments/scripts/create_ground_truth_masks.py
```

This scans `eval_mask/timestep_*/annotations.xml` and writes one `*_mask_ground_truth.png` file per annotated image.

## Run Baseline Color Segmentation

```bash
python experiments/segmentation_experiments/scripts/create_masks_using_colors.py
```

This uses a simple green HSV threshold plus morphological cleanup to create `*_mask_colors.png`.

## Run GroundedSAM Segmentation

The GroundedSAM script expects local checkpoints and the bundled `Grounded-Segment-Anything` sources referenced inside the script:

```bash
python experiments/segmentation_experiments/scripts/create_masks_using_grounded_sam.py
```

By default it writes:

- `*_mask_grounded_sam_vit_b.png`
- `*_mask_grounded_sam_vit_b_overlay_compressed.jpg`

The text prompt, thresholds, and checkpoint paths are configured near the top of the script.

## Evaluate Mask Quality

```bash
python experiments/segmentation_experiments/scripts/analysis/analyze_mask_quality.py
```

This reads the masks in `eval_mask/`, compares each available method against the ground-truth masks, and writes `mask_quality_report_eval_mask.txt`.

The default evaluated methods are:

- `sam3`
- `colors`
- `grounded_sam_vit_b`

## Create Comparison Plots

```bash
python experiments/segmentation_experiments/scripts/analysis/create_plot.py \
  --frame-id frame_1180 \
  --frame-number 1880 \
  --image-index 6
```

This produces a 6-panel figure showing the original image, ground truth, and the different predicted masks for one selected frame.

## Create a Mask Movie

```bash
python experiments/segmentation_experiments/scripts/create_movie_with_masks.py
```

By default this reads images and masks from:

- `images_and_masks/maize_4/images_sam_3_movie`
- `images_and_masks/maize_4/masks_sam_3_movie`

and writes an MP4 to `experiments/segmentation_experiments/movies/`.

## Notes

- The SAM3 workflow was developed in the notebook and is not yet packaged as a single CLI script in this folder.
- Several scripts use fixed top-of-file paths such as `eval_mask`, so they are easiest to run from the repository root unless you edit those constants.
