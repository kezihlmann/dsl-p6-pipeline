# Euler Setup

This is the shortest supported setup path for coworkers on Euler.

## 1. Clone the repo

```bash
cd /cluster/project/cropsci/kzihlmann
git clone https://github.com/kezihlmann/dsl-p6-pipeline.git
cd dsl-p6-pipeline
git submodule update --init --recursive
mkdir -p logs
```

## 2. Open a GPU shell

```bash
srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=8G --time=08:00:00 --pty bash
```

## 3. Load the Euler modules

```bash
module purge
module load stack/2024-06 gcc/12.2.0 cuda/12.1.1 eth_proxy
eval "$(conda shell.bash hook)"
```

Important:

- Use exactly `stack/2024-06 gcc/12.2.0 cuda/12.1.1 eth_proxy` unless you have verified a newer working combination on Euler.
- Do not use the older `stack/2024-05 ... cuda/12.2.1` recipe. On Euler it no longer resolves cleanly with GCC 12, so coworkers will fail before environment creation starts.
- `eth_proxy` is required so Conda and pip can reach `repo.anaconda.com`, `conda.anaconda.org`, and GitHub from the cluster. If you forget it, `conda env create` will fail with `HTTP 000 CONNECTION FAILED`.
- Use a GCC 12.x module for the native CUDA builds in step 5. GCC 13 causes `nvcc` failures on Euler for both `tiny-cuda-nn` and the Wheat-3DGS extensions.

## 4. Create the two conda environments

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
conda env create -f environment-euler.yml
conda env create -f environment-euler-nerfacto.yml
```

If Conda reports connection failures here, first confirm that `eth_proxy` is loaded:

```bash
env | grep -i proxy
```

You should see `http_proxy` and `https_proxy` entries before retrying `conda env create`.

## 5. Finish the one-time native installs

Build `tiny-cuda-nn` for the Nerfacto `tcnn` backend:

```bash
conda activate dsl-p6-nerfacto
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
python scripts/install_tinycudann_euler.py
```

Build the Wheat-3DGS CUDA extensions:

```bash
conda activate dsl-p6-pipeline
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0+PTX"
python scripts/build_wheat_3dgs_extensions.py
```

If your GCC 12 module has a different version name on Euler, load that module instead before running the helpers. The scripts now stop early with a clear error if `CC` and `CXX` resolve to GCC newer than 12.

## 6. Authenticate Hugging Face once

```bash
conda activate dsl-p6-pipeline
hf auth login
```

## 7. Verify the environments

```bash
conda activate dsl-p6-pipeline
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"

conda activate dsl-p6-nerfacto
python -c "import tinycudann, torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
which ns-train
```

## 8. Run the pipeline

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
conda activate dsl-p6-pipeline
python scripts/run_pipeline.py --settings settings_pipeline.txt
```
> [!NOTE]
> For using SAM3 on step 1, accepting the terms of use is required: https://huggingface.co/facebook/sam3

Or submit through Slurm:

```bash
sbatch submit_pipeline.slurm
```

## Data requirements

- `step_1` expects timestep folders with `images/`
- `step_2` additionally requires a provided COLMAP sparse model in `sparse/0/`
- `settings_pipeline.txt` selects:
  - whether `step_1` runs
  - whether `step_2` runs
  - whether `step_2` uses `3dgs` or `nerfacto`

## Update an existing checkout

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
git pull
git submodule update --init --recursive
```

If the environment files changed, recreate or update the affected conda environment and rerun:

```bash
python scripts/install_tinycudann_euler.py
python scripts/build_wheat_3dgs_extensions.py
```
