#!/usr/bin/env bash
set -euo pipefail

FRAME_NAME="${FRAME_NAME:-timestep_3420}"
FRAME_ZIP="${FRAME_ZIP:?Set FRAME_ZIP to a zip containing the frame directory.}"
WHEAT_3DGS_REPO="${WHEAT_3DGS_REPO:?Set WHEAT_3DGS_REPO to a Wheat-3DGS checkout.}"
DATA_ROOT="${DATA_ROOT:-$PWD/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/output}"
RESOLUTION="${RESOLUTION:-4}"
ITERATIONS="${ITERATIONS:-15000}"
TEST_COUNT="${TEST_COUNT:-5}"
LOSS_MODE="${LOSS_MODE:-baseline}"
SILHOUETTE_WEIGHT="${SILHOUETTE_WEIGHT:-0.1}"
BACKGROUND_LOSS_WEIGHT="${BACKGROUND_LOSS_WEIGHT:-0.01}"
EXP_NAME="${EXP_NAME:-${FRAME_NAME}_${LOSS_MODE}_masked}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAME_ROOT="$DATA_ROOT/$FRAME_NAME"
MODEL_DIR="$OUTPUT_ROOT/$EXP_NAME"

python "$SCRIPT_DIR/prepare_masked_colmap_frame.py" \
  --frame-zip "$FRAME_ZIP" \
  --frame-name "$FRAME_NAME" \
  --data-root "$DATA_ROOT" \
  --overwrite

python "$SCRIPT_DIR/patch_wheat_3dgs_for_mask_losses.py" \
  --repo "$WHEAT_3DGS_REPO" \
  --test-count "$TEST_COUNT"

cd "$WHEAT_3DGS_REPO"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PYTHONPATH="$WHEAT_3DGS_REPO:$WHEAT_3DGS_REPO/submodules/simple-knn:$WHEAT_3DGS_REPO/submodules/diff-gaussian-rasterization:$WHEAT_3DGS_REPO/submodules/flashsplat-rasterization:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TORCH_LIB="$("$PYTHON_BIN" - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

rm -rf "$MODEL_DIR"

"$PYTHON_BIN" -u train_mask_loss_3dgs.py \
  -s "$FRAME_ROOT" \
  -m "$MODEL_DIR" \
  --resolution "$RESOLUTION" \
  --iterations "$ITERATIONS" \
  --loss_mode "$LOSS_MODE" \
  --silhouette_weight "$SILHOUETTE_WEIGHT" \
  --background_loss_weight "$BACKGROUND_LOSS_WEIGHT" \
  --test_iterations 7000 "$ITERATIONS" \
  --save_iterations 7000 "$ITERATIONS" \
  --checkpoint_iterations 7000 "$ITERATIONS"

"$PYTHON_BIN" -u render.py \
  -s "$FRAME_ROOT" \
  -m "$MODEL_DIR" \
  --iteration "$ITERATIONS" \
  --resolution "$RESOLUTION"

"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_region_metrics.py" \
  --repo "$WHEAT_3DGS_REPO" \
  --scene "$FRAME_ROOT" \
  --model "$MODEL_DIR" \
  --iteration "$ITERATIONS" \
  --resolution "$RESOLUTION" \
  --out-dir "$MODEL_DIR/metrics"

echo "Done: $MODEL_DIR"
