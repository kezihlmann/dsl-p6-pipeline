#!/bin/bash
#SBATCH --job-name=t1902_hyb_nd
#SBATCH --output=logs/t1902_hyb_nd_%j.out
#SBATCH --error=logs/t1902_hyb_nd_%j.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8

set -euo pipefail

module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy

source "${CONDA_PROFILE:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-wheat3dgs}"

EXPERIMENT_DIR="${EXPERIMENT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCENE="${SCENE:?Set SCENE to the timestep_1902 scene directory.}"
WHEAT_3DGS_REPO="${WHEAT_3DGS_REPO:?Set WHEAT_3DGS_REPO to the Wheat-3DGS checkout.}"
HYBRID_INIT="${HYBRID_INIT:-$EXPERIMENT_DIR/run_metadata/input.ply}"
OUT="${OUT:-$PWD/outputs/t1902_hyb_nd_colmap_nerfacto_nodedup_default}"
ITERATIONS="${ITERATIONS:-15000}"
RESOLUTION="${RESOLUTION:-1}"

mkdir -p "$(dirname "$OUT")" logs

POINTS_PATH="$SCENE/sparse/0/points3D.ply"
BACKUP_PATH="$SCENE/sparse/0/points3D.before_hybrid_init.ply"

if [ ! -f "$HYBRID_INIT" ]; then
  echo "Hybrid initializer not found: $HYBRID_INIT" >&2
  exit 1
fi

if [ ! -f "$POINTS_PATH" ]; then
  echo "Scene points3D.ply not found: $POINTS_PATH" >&2
  exit 1
fi

cp "$POINTS_PATH" "$BACKUP_PATH"
restore_points() {
  if [ -f "$BACKUP_PATH" ]; then
    mv "$BACKUP_PATH" "$POINTS_PATH"
  fi
}
trap restore_points EXIT

cp "$HYBRID_INIT" "$POINTS_PATH"

cd "$WHEAT_3DGS_REPO"

export TORCH_LIB_DIR
TORCH_LIB_DIR="$(python - <<'PY'
import os
import torch
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"
export LD_LIBRARY_PATH="$TORCH_LIB_DIR:${LD_LIBRARY_PATH:-}"

echo "Scene: $SCENE"
echo "Output: $OUT"
echo "Hybrid initializer: $HYBRID_INIT"

python - <<'PY'
import os
import sys
from pathlib import Path

import numpy as np

repo = Path(os.environ["WHEAT_3DGS_REPO"])
scene = Path(os.environ["SCENE"])
sys.path.insert(0, str(repo))

from scene.dataset_readers import fetchPly

pcd = fetchPly(str(scene / "sparse/0/points3D.ply"))
print("initializer_points:", len(pcd.points))
print("has_nan:", np.isnan(pcd.points).any())
print("center:", (pcd.points.min(axis=0) + pcd.points.max(axis=0)) / 2)
print("extent:", pcd.points.max(axis=0) - pcd.points.min(axis=0))

assert len(pcd.points) == 6326
assert not np.isnan(pcd.points).any()
PY

rm -rf "$OUT"

python train_vanilla_3dgs.py \
  -s "$SCENE" \
  -m "$OUT" \
  --iterations "$ITERATIONS" \
  --save_iterations 500 1000 3000 7000 "$ITERATIONS" \
  --test_iterations 500 1000 3000 7000 10000 "$ITERATIONS" \
  --resolution "$RESOLUTION"

python render.py \
  -s "$SCENE" \
  -m "$OUT" \
  --iteration "$ITERATIONS" \
  --resolution "$RESOLUTION"

echo "Done. Output: $OUT"
