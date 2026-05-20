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
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
```

## 4. Create the two conda environments

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
conda env create -f environment-euler.yml
conda env create -f environment-euler-nerfacto.yml
```

## 5. Finish the one-time native installs

Build `tiny-cuda-nn` for the Nerfacto `tcnn` backend:

```bash
conda activate dsl-p6-nerfacto
python scripts/install_tinycudann_euler.py
```

Build the Wheat-3DGS CUDA extensions:

```bash
conda activate dsl-p6-pipeline
python scripts/build_wheat_3dgs_extensions.py
```

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
