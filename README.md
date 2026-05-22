# DSL P6 Pipeline

Clone the repository with submodules:
```bash
git clone https://github.com/kezihlmann/dsl-p6-pipeline.git --recursive
# or, if you already cloned without --recursive:
# git submodule update --init --recursive
```

This repository runs a two-step plant reconstruction pipeline:

1. `step_1`: SAM3 mask generation
2. `step_2`: 3D reconstruction with either `3dgs` or `nerfacto`

The tested target is Euler. Local Linux use is possible with the same two-environment structure, but Euler is the documented path.

For local Linux use, mirror the same split:

- one environment for SAM3 + 3DGS
- one environment for Nerfacto

You can reuse the Euler environment files as a starting point, but you must adapt CUDA, compiler, and native extension installation to your own machine.

## Data Layout

`settings_pipeline.txt` points to an input folder that contains timestep folders such as:

```text
data/maize_4/
  timestep_2004/
    images/
    masks/                  # optional if step_1 will generate masks
    sparse/0/               # required for step_2
```

`step_2` requires a provided COLMAP sparse model in each timestep folder. The pipeline does not run COLMAP feature extraction or mapping.

## Main Entry Point

Run the pipeline with:

```bash
python scripts/run_pipeline.py --settings settings_pipeline.txt
```

`scripts/run_pipeline.py` will:

- run `step_1` in `dsl-p6-pipeline`
- run `step_2` with:
  - `dsl-p6-pipeline` for `3dgs`
  - `dsl-p6-nerfacto` for `nerfacto`

## Euler Setup

Use the short Euler guide in [SETUP_EULER.md](SETUP_EULER.md).

## Important Files

- `settings_pipeline.txt`: pipeline settings
- `environment-euler.yml`: SAM3 + 3DGS environment
- `environment-euler-nerfacto.yml`: Nerfacto environment
- `scripts/build_wheat_3dgs_extensions.py`: builds the Wheat-3DGS CUDA extensions
- `scripts/install_tinycudann_euler.py`: installs `tiny-cuda-nn` for the Nerfacto `tcnn` backend on Euler

## Notes

- `3dgs` depends on `external/Wheat-3DGS`.
- `nerfacto` uses the repository's built-in training and rendering workflow
- Large local experiment outputs under `data/` are ignored by Git.
