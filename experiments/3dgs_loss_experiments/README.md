# 3DGS Mask and Loss Experiments

This folder contains reusable code for the 3DGS experiments that changed Wheat-3DGS to use plant masks and silhouette/alpha losses.

The original work was developed in notebook cells. The scripts here make the same workflow reproducible without editing files by hand.

## Contents

- `scripts/prepare_masked_colmap_frame.py`: normalizes one zipped timestep/frame into the expected Wheat-3DGS layout.
- `scripts/patch_wheat_3dgs_for_mask_losses.py`: patches a Wheat-3DGS checkout so cameras load `masks_binary_active` masks and expose `gt_alpha_mask`.
- `scripts/train_mask_loss_3dgs.py`: training entry point with baseline, masked RGB, and alpha/silhouette loss modes.
- `scripts/run_mask_loss_experiment.sh`: reproducible runner for one frame.
- `scripts/evaluate_region_metrics.py`: computes full-image, plant-only, bbox, and segmentation metrics.
- `scripts/make_side_by_side_panels.py`: makes GT/render/silhouette/SAM3 comparison panels.

## Data Layout

The scripts expect a frame directory like:

```text
timestep_3420/
  images/
  masks_binary_gt/
  masks_binary_active/
  masks_binary_ones/
  sparse/0/
```

`prepare_masked_colmap_frame.py` creates this layout from a zip containing `images`, `masks`, and `sparse`. It preserves the original masks as `masks_binary_gt`, creates `masks_binary_active`, creates `_mask_ground_truth.png` copies from `_mask_sam3.png` masks when needed, and creates all-one masks for ablation runs.

## Patch Wheat-3DGS

```bash
python experiments/3dgs_loss_experiments/scripts/patch_wheat_3dgs_for_mask_losses.py \
  --repo /path/to/Wheat-3DGS \
  --test-count 5
```

The patcher is idempotent. It modifies:

- `scene/dataset_readers.py`: adds `alpha_mask_path`, reads masks from `masks_binary_active`, and uses the last `test_count` cameras as the test set.
- `utils/camera_utils.py`: loads the active mask and passes it into `Camera`.
- `scene/cameras.py`: stores `self.gt_alpha_mask` and masks the ground-truth image.
- `train_mask_loss_3dgs.py`: copied into the Wheat-3DGS repo root.

## Run

```bash
FRAME_ZIP=/path/to/timestep_3420.zip \
FRAME_NAME=timestep_3420 \
WHEAT_3DGS_REPO=/path/to/Wheat-3DGS \
DATA_ROOT=/path/to/data \
OUTPUT_ROOT=/path/to/output \
LOSS_MODE=alpha \
SILHOUETTE_WEIGHT=0.1 \
bash experiments/3dgs_loss_experiments/scripts/run_mask_loss_experiment.sh
```

Useful loss modes:

- `baseline`: standard RGB L1+DSSIM against masked GT.
- `masked_rgb`: computes RGB loss only inside `gt_alpha_mask`.
- `alpha`: standard RGB loss plus BCE on rendered alpha vs mask.
- `rgb_alpha`: masked RGB loss plus BCE on rendered alpha vs mask.
- `foreground_background`: masked foreground RGB plus background alpha penalty.

## Evaluate

```bash
python experiments/3dgs_loss_experiments/scripts/evaluate_region_metrics.py \
  --repo /path/to/Wheat-3DGS \
  --scene /path/to/data/timestep_3420 \
  --model /path/to/output/timestep_3420_alpha \
  --iteration 15000 \
  --out-dir /path/to/metrics/timestep_3420_alpha
```

This writes per-image CSVs plus JSON summaries for full-image, plant-only, bbox, and rendered-silhouette metrics.

Note: For evaluation, 3DGS uses held-out test cameras {2, 6, 10, 14, 18, 21}.
