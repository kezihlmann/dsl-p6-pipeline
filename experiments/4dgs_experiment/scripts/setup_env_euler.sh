#!/bin/bash
set -euo pipefail

EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
REPO=${REPO:-"$EXPERIMENT_ROOT/source/4d-gaussian-splatting"}
CONDA_ENV=${CONDA_ENV:-4d_gaussian_splatting}

module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1
module load eth_proxy

source "$HOME/miniconda3/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda activate "$CONDA_ENV"
else
  conda env create -n "$CONDA_ENV" -f "$REPO/environment.yml"
  conda activate "$CONDA_ENV"
fi

export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"7.5"}
export MAX_JOBS=${MAX_JOBS:-1}

pip install -e "$REPO/simple-knn"
pip install -e "$REPO/pointops2"

echo "Environment ready: $CONDA_ENV"
