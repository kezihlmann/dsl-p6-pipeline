# Euler Setup

This repository uses `external/Wheat-3DGS` as the single official 3DGS dependency.
There is no separate `3dgs_project` runtime dependency in this repo anymore.

This repository now has three implemented stages:

- `scripts/create_sam3_masks.py`: implemented from the notebook template
- `scripts/create_colmap.py`: normalizes a provided COLMAP sparse model into the Wheat-3DGS-ready `sparse/0` layout
- `scripts/create_3dgs_reconstructions.py`: patches Wheat-3DGS for external alpha masks and runs train/render
- `scripts/create_nerfacto_reconstructions.py`: prepares RGBA Nerfstudio datasets from SAM3 masks and runs train/render/export
- `scripts/build_wheat_3dgs_extensions.py`: patches and builds the Wheat-3DGS CUDA extensions from the parent repo
- `scripts/create_video.py`: TODO placeholder
- `scripts/run_pipeline.py`: runs enabled steps from `settings_pipeline.txt`
- `submit_pipeline.slurm`: Slurm entrypoint for Euler
- `environment-euler.yml`: conda environment for SAM3 and 3DGS on Euler
- `environment-euler-nerfacto.yml`: separate conda environment for Nerfacto on Euler

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

## 2. Create the 3DGS environment
Install Miniconda once in your home directory if it is not already available:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

Then request an interactive GPU node before creating the environment:

```bash
srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=8G --time=08:00:00 --pty bash
```

ETH's Euler docs recommend putting persistent module loads in `~/.bashrc`, especially `module load eth_proxy`, and optionally a default software stack such as `module load stack/2024-06`.
For this repository, keep the project commands aligned with `submit_pipeline.slurm`, which currently uses the pinned combination `stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy` for reproducibility.
If you want the proxy available automatically for `wget`, `pip`, or Hugging Face downloads, add this to `~/.bashrc` on Euler and reload your shell once:

```bash
echo 'module load eth_proxy' >> ~/.bashrc
source ~/.bashrc
```

Inside that compute shell, load the same modules you plan to use for jobs and create the 3DGS environment:

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda env create -f environment-euler.yml
conda activate dsl-p6-pipeline
```

The environment file is pinned to `pytorch::pytorch=2.5.1`, `pytorch::torchvision=0.20.1`, `pytorch::torchaudio=2.5.1`, and `pytorch::pytorch-cuda=12.1`.
That is intentional: the PyTorch conda packages are published for CUDA 12.1, while your Euler module stack uses `cuda/12.2.1`.
That combination is normally the practical match on clusters because the loaded CUDA module provides the runtime stack and the 12.1 PyTorch build is compatible with the newer 12.2 driver/runtime environment.
The conda file intentionally avoids `conda-forge` because the broader mixed-channel solve was getting killed on the Euler login node with exit code `137` during metadata resolution.
The heavy ML packages come from `pytorch` and `nvidia`, and the Hugging Face plus Wheat-3DGS runtime packages are installed through `pip` inside the environment.
This environment is intended for `step_1` and the `3dgs` branch of `step_2`.

If the 3DGS environment already exists and you changed dependencies later, update it with:

```bash
conda env update -f environment-euler.yml --prune
```

If `conda` is not on your path after installing Miniconda, source the Miniconda init script first.

If an older failed attempt exists, remove it before recreating the environment:

```bash
conda env remove -n dsl-p6-pipeline
```

To confirm the environment sees the GPU correctly after activation, run:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
conda list | egrep "pytorch|torchvision|torchaudio|libtorch"
```

You want `torch.version.cuda` to report `12.1` and `torch.cuda.is_available()` to be `True`.
If `torch.version.cuda` is `None`, or if `pytorch` / `libtorch` show `cpu_openblas`, the environment is in a broken mixed CPU/CUDA state and the 3DGS CUDA extensions will not build correctly.
In that case, remove and recreate the environment from `environment-euler.yml` before continuing.
The environment files use explicit `pytorch::...` package selectors because Conda can otherwise mix in the CPU `defaults` build of `pytorch` while still installing CUDA `torchvision` and `torchaudio`, which leaves the environment broken.

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

## 2b. Create the Nerfacto environment

Use a separate environment for the Nerfacto branch. This avoids the PyTorch and torchvision conflicts that can appear when Wheat-3DGS and Nerfstudio are mixed into one Euler environment.

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda env create -f environment-euler-nerfacto.yml
conda activate dsl-p6-nerfacto
```

This environment pins:

```text
python=3.11
mkl<2024.1
intel-openmp<2024.1
pytorch::pytorch=2.5.1
pytorch::torchvision=0.20.1
pytorch::torchaudio=2.5.1
pytorch::pytorch-cuda=12.1
setuptools<81
wheel
nerfstudio==1.1.5
av==12.3.0
```

If `dsl-p6-nerfacto` already exists from an older checkout, recreate it or update it so the reference Nerfacto backend dependencies are installed too.

Then install `tiny-cuda-nn` inside that environment on a GPU node:

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda activate dsl-p6-nerfacto
python scripts/install_tinycudann_euler.py
```

This helper detects the visible GPU architecture, sets `TCNN_CUDA_ARCHITECTURES`, and installs the `tiny-cuda-nn` PyTorch binding with the working Euler flags.

Verify the Nerfacto environment with:

```bash
python -c "import tinycudann, torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
which ns-train
ns-train --help | head
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

python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python scripts/build_wheat_3dgs_extensions.py
```

Use `python scripts/build_wheat_3dgs_extensions.py` here instead of `pip install -e ...`.
On Euler, the editable `pip` path can drop CUDA detection during build metadata generation, while the helper script patches the submodule `setup.py` files and builds them directly with `setup.py`.

Any time you recreate or substantially change the `dsl-p6-pipeline` PyTorch stack, rerun `python scripts/build_wheat_3dgs_extensions.py`.

## 6. Run the implemented stages interactively first

Before using Slurm, validate the path layout once:

```bash
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda activate dsl-p6-pipeline
python scripts/create_sam3_masks.py --settings settings_pipeline.txt
python scripts/create_3dgs_reconstructions.py --settings settings_pipeline.txt
```

Your current settings expect data at:

```bash
/cluster/project/cropsci/kzihlmann/dsl-p6-pipeline/data/maize_4
```

Step 2 requires a provided COLMAP sparse model in each selected timestep folder.
If `sparse/0` already exists, it will be reused directly.
If only `sparse/` exists, the reconstruction scripts will normalize it into `sparse/0`.
If no sparse model is present, step 2 will stop and ask you to provide one.

For `reconstruction_method = "3dgs"`, step 2 uses `create_3dgs_reconstructions.py` and writes outputs under each timestep folder in `3dgs-reconstructions/`.
If only SAM3 masks under `masks/` are present, it will automatically prepare a compatible `masks_binary_active` folder for Wheat-3DGS.

For `reconstruction_method = "nerfacto"`, step 2 uses `create_nerfacto_reconstructions.py`.
It prepares `nerfacto-rgba-dataset/` inside each timestep folder, where the SAM3 mask becomes the alpha channel of an RGBA PNG.
The trained Nerfacto outputs, test renders, and exported point cloud are written under each timestep folder in `nerfacto-reconstructions/`.
The Nerfacto branch is aligned with the reference `nerfstudio_project` workflow:
it trains from `images/` with `--downscale-factor 1`, and the optional resolution decrease factor is applied later only during `ns-render`.
After both conda environments exist, `scripts/run_pipeline.py` will automatically hand off step 2 to `dsl-p6-nerfacto` when `reconstruction_method = "nerfacto"`, so the user does not need to switch environments manually between step 1 and step 2.

The automatic handoff only works if both conda environments already exist.
The pipeline always starts from `dsl-p6-pipeline`, and `run_pipeline.py` uses `conda run -n dsl-p6-nerfacto ...` internally for the Nerfacto branch.

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

- `settings_pipeline.txt` can choose the step-2 branch with `reconstruction_method = "3dgs"` or `reconstruction_method = "nerfacto"`.
- Step 2 expects a COLMAP sparse model to already exist; it does not run COLMAP feature extraction or mapping itself.
- The 3DGS branch depends on the Wheat-3DGS submodule and its compiled CUDA extensions.
- The Nerfacto branch should be run from the separate `dsl-p6-nerfacto` environment, not from `dsl-p6-pipeline`.
- The Nerfacto branch depends on `ns-train`, `ns-render`, and `ns-export` being available in that Nerfacto environment.
- The Nerfacto branch is pinned for Euler through `nerfstudio==1.1.5`, follows the reference `nerfstudio_project` training/render workflow, and expects `tiny-cuda-nn` to be installed afterward with `python scripts/install_tinycudann_euler.py` on a GPU node.
- Once both environments are created, `scripts/run_pipeline.py` will automatically use `dsl-p6-pipeline` for SAM3 and 3DGS, and `dsl-p6-nerfacto` for the Nerfacto branch.
- `scripts/run_pipeline.py` is the right place to keep orchestrating the three stages as they are implemented.
- Create the conda environment on a GPU-equipped compute node so the PyTorch and CUDA stack resolve against the same module environment you will use in jobs.
- Use `squeue -u $USER` to monitor submitted jobs.
- There is no separate `requirements-euler.txt` anymore; the conda environment files in `environment-euler.yml` and `environment-euler-nerfacto.yml` are the source of truth for Euler.

## Minimal command sequence

If you just want the shortest working setup path on Euler, this is the sequence:

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=8G --time=08:00:00 --pty bash
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda env create -f environment-euler.yml
conda activate dsl-p6-pipeline
conda env create -f environment-euler-nerfacto.yml
conda activate dsl-p6-nerfacto
python scripts/install_tinycudann_euler.py
conda activate dsl-p6-pipeline
hf auth login
git submodule update --init --recursive
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
conda list | egrep "pytorch|torchvision|torchaudio|libtorch"
export CUDA_HOME=$(dirname "$(dirname "$(which nvcc)")")
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
TORCH_LIB=$(python -c "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
python scripts/build_wheat_3dgs_extensions.py
python scripts/run_pipeline.py --settings settings_pipeline.txt
sbatch submit_pipeline.slurm
squeue -u $USER
```

If you only want to smoke-test the Nerfacto environment itself, use:

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
srun --gpus=1 --cpus-per-task=4 --mem-per-cpu=8G --time=08:00:00 --pty bash
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda env create -f environment-euler-nerfacto.yml
conda activate dsl-p6-nerfacto
python scripts/install_tinycudann_euler.py
python -c "import tinycudann, torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
which ns-train
ns-train --help | head
```
