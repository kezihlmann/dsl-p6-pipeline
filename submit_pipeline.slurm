#!/bin/bash
#SBATCH --job-name=dsl-p6-pipeline
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO_ROOT=/cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
CONDA_ENV_NAME=dsl-p6-pipeline

module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy

if command -v conda >/dev/null 2>&1; then
	eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
	source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
	echo "conda not found. Install Miniconda or adjust submit_pipeline.slurm." >&2
	exit 1
fi

conda activate "$CONDA_ENV_NAME"

cd "$REPO_ROOT"

mkdir -p logs

python scripts/run_pipeline.py --settings settings_pipeline.txt
