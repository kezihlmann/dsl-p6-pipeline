#!/bin/bash
#SBATCH --job-name=4dgs_fg7000
#SBATCH --output=/cluster/project/cropsci/jmercoli/4dgs_project/logs/4dgs_fgweighted_7000_%j.out
#SBATCH --error=/cluster/project/cropsci/jmercoli/4dgs_project/logs/4dgs_fgweighted_7000_%j.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

set -euo pipefail

PROJECT=/cluster/project/cropsci/jmercoli/4dgs_project
REPO=$PROJECT/repos/4d-gaussian-splatting

module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1
module load eth_proxy

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate wheat3dgs

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH

TORCH_LIB_DIR=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
export LD_LIBRARY_PATH=$TORCH_LIB_DIR:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export TORCH_CUDA_ARCH_LIST="7.5"
export MAX_JOBS=1
export TORCH_EXTENSIONS_DIR=$PROJECT/torch_extensions/fgweighted_${SLURM_JOB_ID}
mkdir -p "$TORCH_EXTENSIONS_DIR"

cd "$REPO"

echo "Running foreground-weighted masked 4DGS"
echo "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
echo "Config: configs/plant/close10_sam3mask_fgweighted_7000.yaml"

python train_fgweighted.py --config configs/plant/close10_sam3mask_fgweighted_7000.yaml
