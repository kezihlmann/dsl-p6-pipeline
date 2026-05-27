#!/bin/bash
#SBATCH --job-name=4dgs_test5_3000
#SBATCH --output=/cluster/project/cropsci/jmercoli/4dgs_project/logs/4dgs_test5_3000_%j.out
#SBATCH --error=/cluster/project/cropsci/jmercoli/4dgs_project/logs/4dgs_test5_3000_%j.err
#SBATCH --time=02:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

set -e

cd /cluster/project/cropsci/jmercoli/4dgs_project/repos/4d-gaussian-splatting

module purge
module load stack/2024-06
module load gcc/12.2.0
module load cuda/12.1.1
module load eth_proxy

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate wheat3dgs

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

TORCH_LIB_DIR=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
export LD_LIBRARY_PATH=$TORCH_LIB_DIR:$LD_LIBRARY_PATH

export TORCH_CUDA_ARCH_LIST="7.5"
export CC=$(which gcc)
export CXX=$(which g++)
export MAX_JOBS=1
export WANDB_MODE=disabled

python train.py --config configs/plant/test5_3000.yaml
