# Video Generation Experiments

This folder contains the scripts used to turn aligned Gaussian-splat point clouds into a plant growth video that smoothly interpolates between timesteps while the camera orbits around the plant.

Two renderer variants are included:

- a CPU-oriented version with an optional point-preview mode
- a GPU-oriented version that uses the same inputs and camera logic

## Contents

- `animate_growth_pointclouds.py`: main renderer with CPU/CUDA support, orbit camera controls, optional point-preview mode, and MP4 export.
- `animate_growth_pointclouds_gpu.py`: GPU-focused variant of the same growth-video workflow.
- `download_nerfacto_pointclouds.py`: checks Euler Nerfacto reconstructions and downloads finished `point_cloud.ply` files into `data/point_cloud_XXXX.ply`.
- `stem_axis.txt`: defines a stable stem axis used to align the timesteps consistently before rendering.
- `data/`: example input PLY files for testing the renderer setup.

## Data Layout

The scripts expect a directory of aligned PLY files, for example:

```text
experiments/video_generation/
  data/
    point_cloud_0000.ply
    point_cloud_0378.ply
    point_cloud_0756.ply
    ...
  stem_axis.txt
```

The timestep label is extracted from the trailing digits in each filename.

## Download Point Clouds From Euler

Run from the repository root:

```bash
python experiments/video_generation/download_nerfacto_pointclouds.py
```

By default the script:

- checks timesteps `0000, 0060, ..., 3300`
- looks for Nerfacto outputs under `/cluster/project/cropsci/kzihlmann/dsl-p6-pipeline/data/maize_4`
- expects experiment folders named like `nerfacto_rgba_timestep_0000_down4_30000`
- expects the exported point cloud at `pointcloud_10000/point_cloud.ply`
- downloads each finished result to `experiments/video_generation/data/point_cloud_XXXX.ply`

Useful options:

- `--remote kzihlman@eu-login-16.euler.ethz.ch`
- `--num-iterations 30000`
- `--pointcloud-iteration 10000`
- `--overwrite`

## Render a Growth Video

Run from the repository root:

```bash
python experiments/video_generation/animate_growth_pointclouds.py \
  --data-dir experiments/video_generation/data \
  --axis-file experiments/video_generation/stem_axis.txt \
  --output experiments/video_generation/plant_growth_orbit.mp4
```

The default behavior:

- loads all timestep PLY files from `data/`
- reads the stem axis from `stem_axis.txt`
- cross-fades adjacent timesteps
- moves the camera around the plant during each transition
- writes an MP4 video

## Useful Options

Common options for `animate_growth_pointclouds.py` include:

- `--device auto|cpu|cuda`: choose the rendering device
- `--render-mode splats|points`: switch between full Gaussian splats and a fast point-preview mode
- `--fps`: output frame rate
- `--seconds-per-transition`: duration of one timestep blend
- `--hold-seconds`: pause duration at the first and last timestep
- `--width` and `--height`: output resolution
- `--start-azim`, `--degrees-per-transition`, `--elevation`, `--roll`: camera motion controls
- `--skip-axis-sync`: do not copy the axis coordinates into the PLY headers before rendering

Example preview render:

```bash
python experiments/video_generation/animate_growth_pointclouds.py \
  --render-mode points \
  --device cpu \
  --fps 12 \
  --output experiments/video_generation/plant_growth_preview.mp4
```

## GPU Variant

To run the GPU-oriented script directly:

```bash
python experiments/video_generation/animate_growth_pointclouds_gpu.py \
  --data-dir experiments/video_generation/data \
  --axis-file experiments/video_generation/stem_axis.txt \
  --output experiments/video_generation/plant_growth_orbit.mp4 \
  --device cuda
```

This variant keeps the same core input layout and rendering idea, but is organized around the GPU rendering path.

## Stem Axis File

`stem_axis.txt` stores two points:

- `start=...`
- `end=...`

in the same coordinate system as the PLY files. These points define the stable plant axis used during alignment and rendering.

## Notes

- The bundled `data/` folder is only a small example and not the full experiment dataset.
- The scripts are designed for aligned Gaussian-splat PLY files, not arbitrary raw point clouds.
