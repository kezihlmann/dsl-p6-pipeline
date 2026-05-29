# 4DGS Wheat Experiment

This folder contains a reproducible snapshot of the modified 4D Gaussian Splatting code used for the wheat dynamic reconstruction experiments.

The snapshot is copied from `4dgs_project/repos/4d-gaussian-splatting` and includes the local plant-data changes, SAM3 mask loading, foreground-weighted training, plant configs, rendering helpers, and evaluation/export utilities. Large generated artifacts are intentionally excluded.

## Contents

- `source/4d-gaussian-splatting/`: cleaned source snapshot of the modified 4DGS checkout.
- `source/4d-gaussian-splatting/configs/plant/`: experiment configs for `test5`, `close10`, and `equal10`.
- `scripts/submit_4dgs_euler.sh`: path-configurable Slurm runner for training.
- `scripts/setup_env_euler.sh`: Euler environment/bootstrap helper.
- `environment.yml`: conda environment file with local paths for bundled 4DGS extensions.
- `scripts/materialize_config.py`: writes a runnable config from a template by replacing data/output paths.
- `scripts/compute_*`, `scripts/export_*`, `scripts/render_*`: analysis, export, and render helpers copied from the working project.
- `SOURCE_ORIGIN.txt`: provenance for the copied source snapshot.

## What Is Not Versioned

The following are ignored here: data, trained checkpoints, rendered videos, TensorBoard logs, point-cloud outputs, compiled extension binaries, Python caches, and Slurm logs.

Expected local layout after adding data:

```text
experiments/4dgs_experiment/
  data/
    equal10_dynamic_4dgs/
      timestep_*/
        images/
        masks/
        sparse/0/
  outputs/
  logs/
```

The modified loader also supports a dynamic COLMAP-style root containing `timestep_*` directories. Masks are searched in a sibling `masks/` folder using names such as `<image_stem>_mask_sam3.png`, `<image_stem>.png`, `<image_name>`, `<image_stem>_mask.png`, or `<image_stem>_mask_ground_truth.png`.

## Quick Re-run

These are the minimal steps to rerun the main equal-10 foreground-weighted SAM3-mask experiment on Euler.

1. Go to the experiment folder:

```bash
cd /cluster/project/cropsci/jmercoli/dsl-p6-pipeline/experiments/4dgs_experiment
```

2. Put the dynamic 4DGS dataset under `data/equal10_dynamic_4dgs/`:

```text
data/equal10_dynamic_4dgs/
  timestep_*/
    images/
    masks/
    sparse/0/
```

3. Create or update the conda environment:

```bash
CONDA_ENV=4d_gaussian_splatting \
bash scripts/setup_env_euler.sh
```

4. Submit the main training run:

```bash
sbatch \
  --chdir "$PWD" \
  --export=ALL,CONDA_ENV=4d_gaussian_splatting,SOURCE_PATH="$PWD/data/equal10_dynamic_4dgs",CONFIG_NAME=equal10_sam3mask_fgweighted_7000.yaml,RUN_NAME=equal10_sam3mask_fgweighted_7000 \
  scripts/submit_4dgs_euler.sh
```

5. Check progress and outputs:

```bash
squeue -u "$USER"
tail -f logs/4dgs_4dgs_exp_<JOB_ID>.out
ls outputs/equal10_sam3mask_fgweighted_7000
```

The run writes the exact resolved config to:

```text
outputs/equal10_sam3mask_fgweighted_7000/config_used.yaml
```

6. Resume or continue from the best checkpoint if needed:

```bash
sbatch \
  --chdir "$PWD" \
  --export=ALL,CONDA_ENV=4d_gaussian_splatting,SOURCE_PATH="$PWD/data/equal10_dynamic_4dgs",CONFIG_NAME=equal10_sam3mask_fgweighted_resume7000_nodensify.yaml,RUN_NAME=equal10_resume7000_nodensify,START_CHECKPOINT="$PWD/outputs/equal10_sam3mask_fgweighted_7000/chkpnt_best.pth" \
  scripts/submit_4dgs_euler.sh
```

## Setup On Euler

```bash
cd /cluster/project/cropsci/jmercoli/dsl-p6-pipeline/experiments/4dgs_experiment

CONDA_ENV=4d_gaussian_splatting \
bash scripts/setup_env_euler.sh
```

The original working jobs used an existing `wheat3dgs` conda environment with Euler modules `stack/2024-06`, `gcc/12.2.0`, and `cuda/12.1.1`. The reproducible environment is saved as `environment.yml`, and the upstream source copy also keeps its original `source/4d-gaussian-splatting/environment.yml`.

## Train

Submit the equal-10 foreground-weighted SAM3-mask experiment:

```bash
cd /cluster/project/cropsci/jmercoli/dsl-p6-pipeline/experiments/4dgs_experiment

sbatch \
  --chdir "$PWD" \
  --export=ALL,CONDA_ENV=wheat3dgs,SOURCE_PATH="$PWD/data/equal10_dynamic_4dgs",CONFIG_NAME=equal10_sam3mask_fgweighted_7000.yaml,RUN_NAME=equal10_sam3mask_fgweighted_7000 \
  scripts/submit_4dgs_euler.sh
```

Useful variables:

- `CONFIG_NAME`: a file in `source/4d-gaussian-splatting/configs/plant/`.
- `SOURCE_PATH`: dynamic dataset root.
- `OUTPUT_ROOT`: where run folders are written; default is `outputs/`.
- `RUN_NAME`: output folder name; default is the config name without `.yaml`.
- `CONDA_ENV`: conda environment; default is `wheat3dgs`.
- `START_CHECKPOINT`: optional checkpoint for resume runs.
- `TRAIN_SCRIPT`: default is `train_fgweighted.py`; use `train.py` for non-foreground-weighted configs.

Resume from a previous checkpoint:

```bash
sbatch \
  --chdir "$PWD" \
  --export=ALL,CONDA_ENV=wheat3dgs,SOURCE_PATH="$PWD/data/equal10_dynamic_4dgs",CONFIG_NAME=equal10_sam3mask_fgweighted_resume7000_nodensify.yaml,RUN_NAME=equal10_resume7000_nodensify,START_CHECKPOINT="$PWD/outputs/equal10_sam3mask_fgweighted_7000/chkpnt_best.pth" \
  scripts/submit_4dgs_euler.sh
```

`submit_4dgs_euler.sh` writes the exact config used for each run to `outputs/<RUN_NAME>/config_used.yaml`.

## Notes

The copied historical scripts in `scripts/run_4dgs_*.sh` still contain the absolute paths from the original working area. They are preserved as provenance. For new runs, prefer `scripts/submit_4dgs_euler.sh`, which materializes configs using local paths.

- 4DGS test cameras: `{16, 17, 18, 19, 20, 21}`.
