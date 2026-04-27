# Euler Setup

This repository now has one implemented stage:

- `scripts/create_sam3_masks.py`: implemented from the notebook template
- `scripts/create_colmap.py`: TODO placeholder
- `scripts/create_3dgs_reconstructions.py`: TODO placeholder
- `scripts/create_video.py`: TODO placeholder
- `scripts/run_pipeline.py`: runs enabled steps from `settings_pipeline.txt`
- `submit_pipeline.slurm`: Slurm entrypoint for Euler
- `environment-euler.yml`: conda environment definition for Euler

## 1. Prepare the repository on Euler

```bash
cd /cluster/project/cropsci/kzihlmann
git clone https://github.com/kezihlmann/dsl-p6-pipeline.git
cd dsl-p6-pipeline
mkdir -p logs
```

If the repository is already cloned, update it with:

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
git pull
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

Inside that compute shell, load the same modules you plan to use for jobs and create the conda environment:

```bash
cd /cluster/project/cropsci/kzihlmann/dsl-p6-pipeline
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda env create -f environment-euler.yml
conda activate dsl-p6-pipeline
```

If the environment already exists and you changed dependencies later, update it with:

```bash
conda env update -f environment-euler.yml --prune
```

If `conda` is not on your path after installing Miniconda, source the Miniconda init script first.

## 3. Handle the Hugging Face token safely

Do not hardcode the token in notebooks, scripts, or Slurm files.

Use one of these approaches:

### Option A: Login once on Euler

```bash
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda activate dsl-p6-pipeline
huggingface-cli login
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

## 5. Run the implemented stage interactively first

Before using Slurm, validate the path layout once:

```bash
module purge
module load stack/2024-05 gcc/13.2.0 cuda/12.2.1 eth_proxy
eval "$(conda shell.bash hook)"
conda activate dsl-p6-pipeline
python scripts/create_sam3_masks.py --settings settings_pipeline.txt
```

Your current settings expect data at:

```bash
/cluster/project/cropsci/kzihlmann/dsl-p6-pipeline/data/maize_4
```

and will write masks into each selected timestep folder under `masks/`.

## 6. Submit through Slurm

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

- `settings_pipeline.txt` currently enables only step 1, which is the implemented SAM3 stage.
- Steps 2 to 4 intentionally fail with a TODO message if enabled.
- `scripts/run_pipeline.py` is the right place to keep orchestrating the four stages as they are implemented.
- Create the conda environment on a GPU-equipped compute node so the PyTorch and CUDA stack resolve against the same module environment you will use in jobs.
- Use `squeue -u $USER` to monitor submitted jobs.

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
huggingface-cli login
python scripts/create_sam3_masks.py --settings settings_pipeline.txt
sbatch submit_pipeline.slurm
squeue -u $USER
```