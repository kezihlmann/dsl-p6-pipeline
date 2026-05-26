#!/bin/bash -l
#SBATCH --job-name=dsl-p6-pipeline
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$SCRIPT_DIR}
CONDA_ENV_NAME=${CONDA_ENV_NAME:-dsl-p6-pipeline}
SETTINGS_FILE=${SETTINGS_FILE:-settings_pipeline.txt}
STACK_MODULE=${STACK_MODULE:-stack/2024-06}
GCC_MODULE=${GCC_MODULE:-gcc/12.2.0}
CUDA_MODULE=${CUDA_MODULE:-cuda/12.4.1}
MODULE_ROOT=${MODULE_ROOT:-/cluster/software/lmods}

if ! type module >/dev/null 2>&1; then
	if [ -z "${LMOD_CMD:-}" ]; then
		LMOD_CMD=$(find /cluster/software/stacks -path '*/lmod/lmod/libexec/lmod' -type f 2>/dev/null | head -n 1 || true)
		export LMOD_CMD
	fi

	if [ -n "${LMOD_CMD:-}" ] && [ -x "$LMOD_CMD" ]; then
		module() { eval "$("$LMOD_CMD" sh "$@")"; }
	else
		echo "Could not initialize the module command automatically. Set LMOD_CMD before calling sbatch or adapt submit_pipeline.slurm for this cluster." >&2
		exit 1
	fi
fi

module use "$MODULE_ROOT"
module load "$STACK_MODULE" "$GCC_MODULE" "$CUDA_MODULE" eth_proxy

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

python scripts/run_pipeline.py --settings "$SETTINGS_FILE"
