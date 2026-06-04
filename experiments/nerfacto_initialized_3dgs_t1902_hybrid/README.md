# Nerfacto-Initialized 3DGS Hybrid, Timestep 1902

This is the hybrid 3DGS initialization experiment for timestep 1902.

The initializer combined the original COLMAP point cloud with the Nerfacto-derived point cloud, removed duplicates by rounded coordinates, and used the resulting 6,326-point `input.ply` to initialize Wheat-3DGS.

## Contents

- `source/nerfacto_depth_to_ply.py`: exports a Nerfacto point cloud from model depth/rgb/accumulation outputs.
- `source/align_nerfacto_to_colmap.py`: aligns the raw Nerfstudio-space PLY to COLMAP metric space (uniform scale + translation).
- `source/make_hybrid_colmap_nerfacto_init.py`: combines COLMAP and Nerfacto points and writes the hybrid initializer.
- `source/train_t1902_hybrid_nodedup_default.sbatch`: SLURM recipe used to run the successful 3DGS training job.
- `source/eval_t1902_3dgs_output.py`: evaluation helper used for timestep 1902 3DGS outputs.
- `run_metadata/inputs/`: the two 3,163-point input clouds used to regenerate the hybrid initializer.
- `run_metadata/input.ply`: the archived 6,326-point hybrid initialization cloud.
- `run_metadata/cfg_args`: captured Wheat-3DGS run arguments.
- `run_metadata/cameras.json`: camera metadata from the run output.
- `run_metadata/metrics.summary.json`: plant/full-image metric summary for the run.

## Evaluation camera split

- 3DGS test cameras: `{2, 6, 10, 14, 18, 21}`.
- Nerfacto uses its default code-defined test split.

## Nerfacto-to-COLMAP alignment

The raw Nerfstudio point cloud (output of `nerfacto_depth_to_ply.py` or `ns-export pointcloud`) lives in Nerfstudio's normalized world space.  It must be transformed to COLMAP metric space before being merged with the COLMAP cloud:

```
x_colmap = x_nerfstudio / total_scale + translation
```

For timestep 1902 the alignment parameters are:

| Parameter | Value |
|-----------|-------|
| `total_scale` | **4.51588** |
| `translation` | **[+0.507, −0.518, −0.550]** |

The scale breaks down as `dataparser_scale × extra_factor = 2.4600 × 1.8357 = 4.516`, where `dataparser_scale` comes from `dataparser_transforms.json` of the trained Nerfacto model and the extra factor accounts for additional internal Nerfstudio normalization.  The translation was determined empirically so that the scaled Nerfstudio cloud overlaps with the COLMAP scene.

To reproduce `run_metadata/inputs/nerfacto_3163_axis_aligned_to_colmap_t1902_wheatfmt.ply` from the raw export:

```bash
python experiments/nerfacto_initialized_3dgs_t1902_hybrid/source/align_nerfacto_to_colmap.py \
  --nerfacto-ply  <raw_nerfacto_export.ply> \
  --out           /tmp/nerfacto_aligned.ply \
  --scale         4.51588 \
  --translation   0.507 -0.518 -0.550
```

For a different timestep, supply `--dataparser-transforms` to read the scale automatically and `--colmap-ply` to centroid-match the translation (may need manual refinement):

```bash
python experiments/nerfacto_initialized_3dgs_t1902_hybrid/source/align_nerfacto_to_colmap.py \
  --nerfacto-ply           <raw_export.ply> \
  --out                    <aligned_output.ply> \
  --dataparser-transforms  <nerfacto_run_dir>/dataparser_transforms.json \
  --extra-scale            1.836 \
  --colmap-ply             <timestep_dir>/sparse/0/points3D.ply
```

## Reproduce the initializer

Run from the repository root in an environment with `numpy` and `plyfile`:

```bash
python experiments/nerfacto_initialized_3dgs_t1902_hybrid/source/make_hybrid_colmap_nerfacto_init.py \
  --colmap-ply experiments/nerfacto_initialized_3dgs_t1902_hybrid/run_metadata/inputs/colmap_points3D_original_3163.ply \
  --nerfacto-ply experiments/nerfacto_initialized_3dgs_t1902_hybrid/run_metadata/inputs/nerfacto_3163_axis_aligned_to_colmap_t1902_wheatfmt.ply \
  --out /tmp/t1902_hybrid_input.ply \
  --voxel 0.005
```

Expected output:

```text
COLMAP points: 3163
Nerfacto points: 3163
Hybrid points: 6326
```

To compare against the archived initializer:

```bash
sha256sum experiments/nerfacto_initialized_3dgs_t1902_hybrid/run_metadata/input.ply
sha256sum /tmp/t1902_hybrid_input.ply
```

The archived initializer checksum is:

```text
0aea3a55c5d2656a0a6a72920c138a1684f878e28335e163376e03d0a0b82390
```

## Reproduce the 3DGS run

The training job expects the timestep 1902 scene folder and a Wheat-3DGS checkout. On Euler, submit it with:

```bash
EXPERIMENT_DIR=$PWD/experiments/nerfacto_initialized_3dgs_t1902_hybrid \
SCENE=/path/to/timestep_1902 \
WHEAT_3DGS_REPO=/path/to/Wheat-3DGS \
OUT=/path/to/outputs/t1902_hyb_nd_colmap_nerfacto_nodedup_default \
sbatch experiments/nerfacto_initialized_3dgs_t1902_hybrid/source/train_t1902_hybrid_nodedup_default.sbatch
```

By default the job uses `run_metadata/input.ply`, trains for 15,000 iterations, renders iteration 15,000, checks that the initializer has exactly 6,326 points, and restores the scene's original `sparse/0/points3D.ply` on exit.

To use a freshly regenerated initializer, set:

```bash
HYBRID_INIT=/tmp/t1902_hybrid_input.ply
```

## Checksums

```text
0aea3a55c5d2656a0a6a72920c138a1684f878e28335e163376e03d0a0b82390  run_metadata/input.ply
2f0bb47b04c72d6c8a3f42c8375909cc1ac96cf5d304cb2f45637b22b3883246  run_metadata/inputs/colmap_points3D_original_3163.ply
8a736ba946e2eedbfb186761d7830c00a46204da3eaefbe67ec4a096b5c6a9f9  run_metadata/inputs/nerfacto_3163_axis_aligned_to_colmap_t1902_wheatfmt.ply
b0f70a14f0a89f6faf8a78db5b0b782849d19d5bf2f5b3782c63c050a2772d1f  run_metadata/metrics.summary.json
```

## Notes

- Initial hybrid cloud: 6,326 vertices.
- Final iteration 15000 point cloud in the local output: 52,747 vertices.
- Mean full PSNR from the archived summary: 34.72.
- Mean plant PSNR from the archived summary: 17.87.
- Mean silhouette IoU from the archived summary: 0.824.

- For the test cameras: Nerfacto uses its default code-defined test split.
