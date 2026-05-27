#!/bin/bash
#SBATCH --job-name=4dgs_exp
#SBATCH --output=logs/4dgs_%x_%j.out
#SBATCH --error=logs/4dgs_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

set -euo pipefail

EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REPO=${REPO:-"$EXPERIMENT_ROOT/source/4d-gaussian-splatting"}
CONDA_ENV=${CONDA_ENV:-wheat3dgs}

CONFIG_NAME=${CONFIG_NAME:-equal10_sam3mask_fgweighted_7000.yaml}
CONFIG_TEMPLATE=${CONFIG_TEMPLATE:-"$REPO/configs/plant/$CONFIG_NAME"}
RUN_NAME=${RUN_NAME:-${CONFIG_NAME%.yaml}}
SOURCE_PATH=${SOURCE_PATH:-"$EXPERIMENT_ROOT/data/equal10_dynamic_4dgs"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"$EXPERIMENT_ROOT/outputs"}
MODEL_PATH=${MODEL_PATH:-"$OUTPUT_ROOT/$RUN_NAME"}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-train_fgweighted.py}
START_CHECKPOINT=${START_CHECKPOINT:-}
LOADED_PTH=${LOADED_PTH:-}

mkdir -p "$EXPERIMENT_ROOT/logs" "$MODEL_PATH"

module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1
module load eth_proxy

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PATH="$CUDA_HOME/bin:$PATH"

TORCH_LIB_DIR=$(python - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.5"}
export MAX_JOBS=${MAX_JOBS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-"$EXPERIMENT_ROOT/torch_extensions/${RUN_NAME}_${SLURM_JOB_ID:-local}"}
mkdir -p "$TORCH_EXTENSIONS_DIR"

CONFIG_USED="$MODEL_PATH/config_used.yaml"
python "$EXPERIMENT_ROOT/scripts/materialize_config.py" \
  --template "$CONFIG_TEMPLATE" \
  --output "$CONFIG_USED" \
  --source-path "$SOURCE_PATH" \
  --model-path "$MODEL_PATH" \
  --loaded-pth "$LOADED_PTH"

cd "$REPO"

CMD=(python "$TRAIN_SCRIPT" --config "$CONFIG_USED")
if [[ -n "$START_CHECKPOINT" ]]; then
  CMD+=(--start_checkpoint "$START_CHECKPOINT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
