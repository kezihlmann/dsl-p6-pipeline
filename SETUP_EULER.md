# Euler Setup

This repository uses `external/Wheat-3DGS` as the single official 3DGS dependency.
There is no separate `3dgs_project` runtime dependency in this repo anymore.

This repository now has three implemented stages:

- `scripts/create_sam3_masks.py`: implemented from the notebook template
- `scripts/create_colmap.py`: creates a COLMAP sparse model from `images` when missing, or normalizes an existing sparse model into the Wheat-3DGS-ready `sparse/0` layout
- `scripts/create_3dgs_reconstructions.py`: patches Wheat-3DGS for external alpha masks and runs train/render
- `scripts/create_video.py`: TODO placeholder
- `scripts/run_pipeline.py`: runs enabled steps from `settings_pipeline.txt`
- `submit_pipeline.slurm`: Slurm entrypoint for Euler
- `environment-euler.yml`: conda environment definition for Euler

## 1. Prepare the repository on Euler

```bash
cd /cluster/project/cropsci/kzihlmann
git clone https://github.com/kezihlmann/dsl-p6-pipeline.git
cd dsl-p6-pipeline
git submodule update --init --recursive
mkdir -p logs
```

If the repository is already cloned, update it with:

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
git pull
git submodule update --init --recursive
```

## 2. Create a Python environment
Install Miniconda once in your home directory if it is not already available:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Then request an interactive GPU node before creating the environment:

```bash
srun --gpus=1 --pty bash
```

ETH's Euler docs recommend putting persistent module loads in `~/.bashrc`, especially `module load eth_proxy`, and optionally a default software stack such as `module load stack/2024-06`.
For this repository, keep the project commands aligned with `submit_pipeline.slurm`, which currently uses the pinned combination `stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy` for reproducibility.
If you want the proxy available automatically for `wget`, `pip`, or Hugging Face downloads, add this to `~/.bashrc` on Euler and reload your shell once:

```bash
echo 'module load eth_proxy' >> ~/.bashrc
source ~/.bashrc
```

Inside that compute shell, load the same modules you plan to use for jobs and create the conda environment:

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda env create -f environment-euler.yml
conda activate dsl-p6-pipeline
```

The environment file is pinned to `pytorch=2.5.*`, `torchvision=0.20.*`, and `pytorch-cuda=12.1`.
That is intentional: the PyTorch conda packages are published for CUDA 12.1, while your Euler module stack uses `cuda/12.2.1`.
That combination is normally the practical match on clusters because the loaded CUDA module provides the runtime stack and the 12.1 PyTorch build is compatible with the newer 12.2 driver/runtime environment.
The conda file intentionally avoids `conda-forge` because the broader mixed-channel solve was getting killed on the Euler login node with exit code `137` during metadata resolution.
The heavy ML packages come from `pytorch` and `nvidia`, COLMAP is installed through `conda-forge`, and the Hugging Face plus Wheat-3DGS runtime packages are installed through `pip` inside the environment.

If the environment already exists and you changed dependencies later, update it with:

```bash
conda env update -f environment-euler.yml --prune
```

After activation, confirm COLMAP is available:

```bash
which colmap
colmap -h | head
```

If `conda` is not on your path after installing Miniconda, source the Miniconda init script first.

If an older failed attempt exists, remove it before recreating the environment:

```bash
conda env remove -n dsl-p6-pipeline
```

To confirm the environment sees the GPU correctly after activation, run:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If the environment solve still gets killed on the login node, request a GPU shell first and run the same create command there:

```bash
srun --gpus=1 --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
unset PYTHONHOME
unset PYTHONPATH
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment-euler.yml
```

## 3. Handle the Hugging Face token safely

Do not hardcode the token in notebooks, scripts, or Slurm files.

Use one of these approaches:

### Option A: Login once on Euler

```bash
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda activate dsl-p6-pipeline
hf auth login
```

This stores the token in your Hugging Face cache under your account on Euler.
After that, `create_sam3_masks.py` can use the cached token automatically.

### Option B: Export a shell environment variable

```bash
export HF_TOKEN=your_token_here
```

If you submit through Slurm and want the current shell environment exported:

```bash
sbatch --export=ALL submit_pipeline.slurm
```

Use this only in your shell session or in a private, ignored file that you source manually.
Do not commit the token.

## 4. Pre-download the model once

This avoids paying the download cost inside the first compute job:

```bash
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda activate dsl-p6-pipeline
python - <<'PY'
from os import getenv
from transformers import Sam3Model, Sam3Processor

token = getenv("HF_TOKEN") or getenv("HUGGING_FACE_HUB_TOKEN")
kwargs = {"token": token} if token else {}
Sam3Processor.from_pretrained("facebook/sam3", **kwargs)
Sam3Model.from_pretrained("facebook/sam3", **kwargs)
print("SAM3 model cache is ready.")
PY
```

## 5. Build the Wheat-3DGS CUDA extensions once

After the environment is active and the submodule is checked out, build the three native extensions inside a GPU shell:

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
git submodule update --init --recursive

export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"

pip install -v --no-build-isolation -e external/Wheat-3DGS/submodules/simple-knn
pip install -v --no-build-isolation -e external/Wheat-3DGS/submodules/diff-gaussian-rasterization
pip install -v --no-build-isolation -e external/Wheat-3DGS/submodules/flashsplat-rasterization
```

On Euler's `stack/2024-05 gcc/13.2.0 cuda/12.2.1` combination, CUDA may reject the host compiler version while building the last two extensions. If that happens, patch both extension `setup.py` files to add `-allow-unsupported-compiler` to the `nvcc` compile args, remove their `build/` directories, and rerun those two `pip install -e ...` commands.

## 6. Run the implemented stages interactively first

Before using Slurm, validate the path layout once:

```bash
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda activate dsl-p6-pipeline
python scripts/create_sam3_masks.py --settings settings_pipeline.txt
python scripts/create_colmap.py --settings settings_pipeline.txt
python scripts/create_3dgs_reconstructions.py --settings settings_pipeline.txt
```

Your current settings expect data at:

```bash
/cluster/project/cropsci/kzihlmann/dsl-p6-pipeline/data/maize_4
```

Step 3 will use `masks_binary_active` if it already exists.
If only SAM3 masks under `masks/` are present, `create_3dgs_reconstructions.py` will automatically prepare a compatible `masks_binary_active` folder for Wheat-3DGS.
If `sparse/0` is missing, step 2 will automatically run `create_colmap.py` first to build the sparse COLMAP model from the timestep `images` folder.

## 7. Submit through Slurm

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda activate dsl-p6-pipeline
sbatch submit_pipeline.slurm
```

Check job status with:

```bash
squeue -u $USER
tail -f logs/dsl-p6-pipeline-<jobid>.out
```

## Notes

- `settings_pipeline.txt` currently enables only step 1 by default; enable steps 2 and 3 when you want the full reconstruction run through `scripts/run_pipeline.py`.
- Step 2 is dataset normalization for Wheat-3DGS, not a standalone COLMAP feature extraction/mapping stage.
- Step 3 depends on the Wheat-3DGS submodule and its compiled CUDA extensions.
- `scripts/run_pipeline.py` is the right place to keep orchestrating the four stages as they are implemented.
- Create the conda environment on a GPU-equipped compute node so the PyTorch and CUDA stack resolve against the same module environment you will use in jobs.
- Use `squeue -u $USER` to monitor submitted jobs.
- There is no separate `requirements-euler.txt` anymore; the conda environment in `environment-euler.yml` is the single source of truth for Euler.

## Minimal command sequence

If you just want the shortest working setup path on Euler, this is the sequence:

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
srun --gpus=1 --pty bash
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda env create -f environment-euler.yml
conda activate dsl-p6-pipeline
hf auth login
git submodule update --init --recursive
python scripts/create_sam3_masks.py --settings settings_pipeline.txt
python scripts/create_colmap.py --settings settings_pipeline.txt
python scripts/create_3dgs_reconstructions.py --settings settings_pipeline.txt
sbatch submit_pipeline.slurm
squeue -u $USER
```
