from __future__ import annotations

import argparse
import ast
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


TIMESTEP_PATTERN = re.compile(r"timestep_(\d+)$")
FIXED_TEST_INDICES = (2, 6, 10, 14, 18, 21)
DEFAULT_SAVE_ITERATIONS = (7000,)
DEFAULT_TEST_ITERATIONS = (1000, 3000, 7000, 10000)


@dataclass
class Settings:
    input_folder: str
    first_timestep: int
    last_timestep: int
    num_timesteps: int
    resolution_decrease_factor: int
    num_iterations: int


def parse_settings(settings_path: Path) -> Settings:
    values: dict[str, object] = {}
    for raw_line in settings_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        try:
            values[key] = ast.literal_eval(raw_value)
        except (SyntaxError, ValueError):
            values[key] = raw_value.strip('"')

    return Settings(
        input_folder=str(values["input_folder"]),
        first_timestep=int(values["first_timestep"]),
        last_timestep=int(values["last_timestep"]),
        num_timesteps=int(values.get("num_timesteps", values.get("number_of_timesteps", 1))),
        resolution_decrease_factor=int(values.get("resolution_decrease_factor", 1)),
        num_iterations=int(values.get("num_iterations", values.get("number_of_iteration", 15000))),
    )


def find_available_timesteps(input_root: Path) -> list[tuple[int, Path]]:
    timesteps: list[tuple[int, Path]] = []
    for path in sorted(input_root.iterdir()):
        if not path.is_dir():
            continue
        match = TIMESTEP_PATTERN.match(path.name)
        if match:
            timesteps.append((int(match.group(1)), path))
    return timesteps


def choose_timesteps(available: list[tuple[int, Path]], settings: Settings) -> list[tuple[int, Path]]:
    filtered = [
        item
        for item in available
        if settings.first_timestep <= item[0] <= settings.last_timestep
    ]
    if not filtered:
        raise ValueError(
            f"No timestep folders found between {settings.first_timestep} and {settings.last_timestep}."
        )

    requested = min(settings.num_timesteps, len(filtered))
    if requested <= 0:
        raise ValueError("num_timesteps must be at least 1.")
    if requested == len(filtered):
        return filtered
    if requested == 1:
        return [filtered[0]]

    span = len(filtered) - 1
    selected_indices = sorted(
        {
            int(round((position * span) / (requested - 1)))
            for position in range(requested)
        }
    )
    while len(selected_indices) < requested:
        for index in range(len(filtered)):
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) == requested:
                break
    return [filtered[index] for index in sorted(selected_indices)]


def patch_dataset_readers(repo_dir: Path) -> bool:
    path = repo_dir / "scene" / "dataset_readers.py"
    text = path.read_text(encoding="utf-8")
    original = text

    if "alpha_mask_path: str | None" not in text and "alpha_mask_path: str" not in text:
        text = text.replace(
            "    mask_paths: List[str]\n",
            "    mask_paths: List[str]\n    alpha_mask_path: str | None\n",
        )

    pattern = re.compile(
        r'        image_path = os\.path\.join\(images_folder, os\.path\.basename\(extr\.name\)\)\n'
        r'        image_name = os\.path\.basename\(image_path\)\.split\("\."\)\[0\]\n'
        r'        image = Image\.open\(image_path\)\n'
        r'(?:\n        alpha_mask_path = .*?\n(?:        .*?\n)*)?',
        re.DOTALL,
    )
    replacement = (
        "        image_path = os.path.join(images_folder, os.path.basename(extr.name))\n"
        "        image_name = os.path.basename(image_path).split(\".\")[0]\n"
        "        image = Image.open(image_path)\n"
        "\n"
        "        alpha_mask_path = os.path.join(\n"
        "            os.path.dirname(images_folder),\n"
        "            \"masks_binary_active\",\n"
        "            image_name + \"_mask_ground_truth.png\"\n"
        "        )\n"
        "        alpha_mask_path = os.path.normpath(alpha_mask_path)\n"
        "        if not os.path.exists(alpha_mask_path):\n"
        "            alpha_mask_path = None\n"
    )
    text, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise ValueError("Failed to patch alpha-mask source in dataset_readers.py")

    if "alpha_mask_path=alpha_mask_path" not in text:
        text = text.replace(
            "        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,\n"
            "                              image_path=image_path, image_name=image_name, width=width, height=height,\n"
            "                              bbox_path=bbox_path, mask_paths=mask_paths)\n",
            "        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,\n"
            "                              image_path=image_path, image_name=image_name, width=width, height=height,\n"
            "                              bbox_path=bbox_path, mask_paths=mask_paths, alpha_mask_path=alpha_mask_path)\n",
        )

    if "test_indices = {2, 6, 10, 14, 18, 21}" not in text:
        eval_pattern = re.compile(
            r"    if eval:\n(?:.*\n)*?    else:\n        train_cam_infos = cam_infos\n        test_cam_infos = \[\]\n",
            re.MULTILINE,
        )
        eval_replacement = (
            "    if eval:\n"
            "        test_indices = {2, 6, 10, 14, 18, 21}\n"
            "        train_cam_infos = []\n"
            "        test_cam_infos = []\n"
            "        for idx, cam_info in enumerate(cam_infos):\n"
            "            if idx in test_indices:\n"
            "                test_cam_infos.append(cam_info)\n"
            "            else:\n"
            "                train_cam_infos.append(cam_info)\n"
            "        print(f\"Train Cam list with {len(train_cam_infos)} cams: {[cam.image_name for cam in train_cam_infos]}\")\n"
            "        print(f\"Test Cam list with {len(test_cam_infos)} cams: {[cam.image_name for cam in test_cam_infos]}\")\n"
            "    else:\n"
            "        train_cam_infos = cam_infos\n"
            "        test_cam_infos = []\n"
        )
        text, count = eval_pattern.subn(eval_replacement, text, count=1)
        if count != 1:
            raise ValueError("Failed to patch fixed train/test split in dataset_readers.py")

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_camera_utils(repo_dir: Path) -> bool:
    path = repo_dir / "utils" / "camera_utils.py"
    text = path.read_text(encoding="utf-8")
    original = text

    if "from PIL import Image" not in text:
        text = text.replace("import numpy as np\n", "import numpy as np\nfrom PIL import Image\n")
    if "import torch\n" not in text:
        text = text.replace("from PIL import Image\n", "from PIL import Image\nimport torch\n")

    if 'getattr(cam_info, "alpha_mask_path", None)' not in text:
        pattern = re.compile(
            r"    gt_image = resized_image_rgb\[:3, \.\.\.\]\n"
            r"    loaded_mask = None\n\n"
            r"    if resized_image_rgb\.shape\[(?:0|1)\] == 4:\n"
            r"        loaded_mask = resized_image_rgb\[3:4, \.\.\.\]\n",
        )
        replacement = (
            "    gt_image = resized_image_rgb[:3, ...]\n"
            "    loaded_mask = None\n\n"
            "    if resized_image_rgb.shape[1] == 4:\n"
            "        loaded_mask = resized_image_rgb[3:4, ...]\n\n"
            "    if getattr(cam_info, \"alpha_mask_path\", None) is not None:\n"
            "        alpha_pil = Image.open(cam_info.alpha_mask_path).convert(\"L\")\n"
            "        alpha_resized = alpha_pil.resize(resolution, Image.NEAREST)\n"
            "        alpha_np = np.array(alpha_resized)\n"
            "        alpha_np = (alpha_np > 128).astype(np.float32)\n"
            "        loaded_mask = torch.from_numpy(alpha_np).unsqueeze(0)\n"
        )
        text, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise ValueError("Failed to patch loadCam mask handling in camera_utils.py")

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_cameras(repo_dir: Path) -> bool:
    path = repo_dir / "scene" / "cameras.py"
    text = path.read_text(encoding="utf-8")
    original = text

    if "self.original_image *= self.gt_alpha_mask" not in text:
        pattern = re.compile(
            r"        if gt_alpha_mask is not None:\n"
            r"            self\.original_image \*= gt_alpha_mask\.to\(self\.data_device\)\n"
            r"        else:\n"
            r"            self\.original_image \*= torch\.ones\(\(1, self\.image_height, self\.image_width\), device=self\.data_device\)\n",
        )
        replacement = (
            "        if gt_alpha_mask is not None:\n"
            "            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)\n"
            "        else:\n"
            "            self.gt_alpha_mask = torch.ones((1, self.image_height, self.image_width), device=self.data_device)\n"
            "\n"
            "        self.original_image *= self.gt_alpha_mask\n"
        )
        text, count = pattern.subn(replacement, text, count=1)
        if count != 1:
            raise ValueError("Failed to patch alpha-mask block in cameras.py")

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_train_vanilla_3dgs(repo_dir: Path) -> bool:
    path = repo_dir / "train_vanilla_3dgs.py"
    text = path.read_text(encoding="utf-8")
    original = text

    if 'dir="/cluster/project/cropsci/jmercoli/3dgs_project/logs/wandb"' in text:
        text = text.replace(
            'wandb.init(project="Wheat-GS", name=dataset.source_path.split("/")[-1], dir="/cluster/project/cropsci/jmercoli/3dgs_project/logs/wandb", mode="disabled")',
            'wandb.init(project="Wheat-GS", name=dataset.source_path.split("/")[-1], mode="disabled")',
        )

    old_import = "import os\nimport wandb\nimport torch\n"
    new_import = (
        "import os\n"
        "try:\n"
        "    import wandb\n"
        "except ImportError:\n"
        "    class _WandbStub:\n"
        "        @staticmethod\n"
        "        def init(*args, **kwargs):\n"
        "            return None\n"
        "\n"
        "        @staticmethod\n"
        "        def log(*args, **kwargs):\n"
        "            return None\n"
        "\n"
        "    wandb = _WandbStub()\n"
        "import torch\n"
    )
    if "class _WandbStub:" not in text:
        if old_import not in text:
            raise ValueError("Failed to find wandb import block in train_vanilla_3dgs.py")
        text = text.replace(old_import, new_import, 1)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_extension_setup_files(repo_dir: Path) -> list[str]:
    changed: list[str] = []
    targets = [
        repo_dir / "submodules" / "simple-knn" / "setup.py",
        repo_dir / "submodules" / "diff-gaussian-rasterization" / "setup.py",
        repo_dir / "submodules" / "flashsplat-rasterization" / "setup.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        original = text
        if '"-allow-unsupported-compiler"' not in text:
            text = text.replace(
                'extra_compile_args={"nvcc": [',
                'extra_compile_args={"nvcc": ["-allow-unsupported-compiler", ',
            )
            text = text.replace(
                'extra_compile_args={"nvcc": [], "cxx": cxx_compiler_flags}',
                'extra_compile_args={"nvcc": ["-allow-unsupported-compiler"], "cxx": cxx_compiler_flags}',
            )
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed.append(str(path.relative_to(repo_dir)))

    return changed


def patch_wheat_3dgs(repo_dir: Path) -> list[str]:
    if not repo_dir.exists():
        raise FileNotFoundError(f"Wheat-3DGS repository not found: {repo_dir}")

    changed: list[str] = []
    if patch_dataset_readers(repo_dir):
        changed.append("scene/dataset_readers.py")
    if patch_camera_utils(repo_dir):
        changed.append("utils/camera_utils.py")
    if patch_cameras(repo_dir):
        changed.append("scene/cameras.py")
    if patch_train_vanilla_3dgs(repo_dir):
        changed.append("train_vanilla_3dgs.py")
    changed.extend(patch_extension_setup_files(repo_dir))
    return changed


def ensure_sparse_zero(frame_root: Path) -> None:
    sparse_zero = frame_root / "sparse" / "0"
    if not sparse_zero.exists():
        raise FileNotFoundError(
            f"Missing COLMAP sparse model for {frame_root.name} at {frame_root}. "
            "Step 2 requires a provided sparse model before 3DGS reconstruction can start. "
            "Expected either 'sparse/0' or a 'sparse' folder that can be normalized into 'sparse/0'."
        )


def normalize_masks_binary_active(frame_root: Path) -> Path:
    active_dir = frame_root / "masks_binary_active"
    if active_dir.exists():
        return active_dir

    source_dir = None
    rename_mode = None
    if (frame_root / "masks").exists():
        source_dir = frame_root / "masks"
        rename_mode = "sam3"
    elif (frame_root / "masks_binary_gt").exists():
        source_dir = frame_root / "masks_binary_gt"
        rename_mode = "copy"
    elif (frame_root / "masks_binary").exists():
        source_dir = frame_root / "masks_binary"
        rename_mode = "copy"

    if source_dir is None:
        raise FileNotFoundError(
            f"No usable mask directory found for {frame_root}. Expected one of masks_binary_active, masks, masks_binary_gt, or masks_binary."
        )

    active_dir.mkdir(parents=False, exist_ok=False)
    for source_path in sorted(source_dir.glob("*.png")):
        if rename_mode == "sam3":
            stem = source_path.stem
            if not stem.endswith("_mask_sam3"):
                continue
            image_stem = stem[: -len("_mask_sam3")]
            target_name = f"{image_stem}_mask_ground_truth.png"
        else:
            target_name = source_path.name
        shutil.copy2(source_path, active_dir / target_name)

    if not any(active_dir.iterdir()):
        raise FileNotFoundError(f"No masks were prepared in {active_dir}")

    return active_dir


def ensure_prepared_timestep(frame_root: Path) -> Path:
    ensure_sparse_zero(frame_root)
    return normalize_masks_binary_active(frame_root)


def selected_timesteps_need_sparse_normalization(frame_roots: list[Path]) -> bool:
    return any((frame_root / "sparse").exists() and not (frame_root / "sparse" / "0").exists() for frame_root in frame_roots)


def run_sparse_normalization(settings_path: Path, repo_root: Path, dry_run: bool) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "create_colmap.py"),
        "--settings",
        str(settings_path),
    ]
    print("Some selected timesteps already have a sparse model but are missing sparse/0. Running sparse normalization first.")
    print(f"Running: {' '.join(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=repo_root, check=True)


def resolution_label(resolution_decrease_factor: int) -> str:
    return "fullres" if resolution_decrease_factor == 1 else f"res{resolution_decrease_factor}"


def build_model_dir(frame_root: Path, settings: Settings) -> Path:
    folder_name = (
        f"{frame_root.name}_masked_"
        f"{resolution_label(settings.resolution_decrease_factor)}_"
        f"16train6test_{settings.num_iterations}"
    )
    return frame_root / "3dgs-reconstructions" / folder_name


def build_environment(repo_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    python_path_entries = [str(repo_dir)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        python_path_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    return env


def select_iterations(total_iterations: int, defaults: tuple[int, ...]) -> list[int]:
    result = {total_iterations}
    for value in defaults:
        if value < total_iterations:
            result.add(value)
    return sorted(result)


def run_command(command: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print(f"Running: {' '.join(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, env=env, check=True)


def verify_masked_scene(
    repo_dir: Path,
    frame_root: Path,
    model_dir: Path,
    resolution_decrease_factor: int,
    env: dict[str, str],
    dry_run: bool,
) -> None:
    debug_model_dir = model_dir / "debug_check"
    if not dry_run:
        debug_model_dir.mkdir(parents=True, exist_ok=True)
    inline_code = (
        "from argparse import ArgumentParser\n"
        "from arguments import ModelParams\n"
        "from scene import Scene\n"
        "from scene.gaussian_model import GaussianModel\n"
        "parser = ArgumentParser()\n"
        "lp = ModelParams(parser)\n"
        "args = parser.parse_args([\n"
        f"    '-s', r'{frame_root}',\n"
        f"    '-m', r'{debug_model_dir}',\n"
        f"    '--resolution', '{resolution_decrease_factor}',\n"
        "    '--eval',\n"
        "])\n"
        "scene = Scene(args, GaussianModel(3), shuffle=False)\n"
        "train = scene.getTrainCameras()\n"
        "test = scene.getTestCameras()\n"
        "print('TRAIN:', len(train))\n"
        "print('TEST:', len(test))\n"
        "assert len(train) == 16, f'Expected 16 train cameras, got {len(train)}'\n"
        "assert len(test) == 6, f'Expected 6 test cameras, got {len(test)}'\n"
        "cam = train[0]\n"
        "img = cam.original_image.detach().cpu()\n"
        "mask = cam.gt_alpha_mask.detach().cpu()\n"
        "outside = img[:, mask[0] < 0.5]\n"
        "print('Mask outside max:', float(outside.max()) if outside.numel() else None)\n"
        "assert outside.numel() == 0 or float(outside.max()) == 0.0, 'Background is not masked to zero'\n"
    )
    run_command([sys.executable, "-c", inline_code], repo_dir, env, dry_run)


def run(settings_path: Path, repo_dir: Path, dry_run: bool, overwrite: bool) -> int:
    settings = parse_settings(settings_path)
    repo_root = settings_path.parent
    input_root = (repo_root / settings.input_folder).resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")
    available_timesteps = find_available_timesteps(input_root)
    selected_timesteps = choose_timesteps(available_timesteps, settings)
    patched_files = patch_wheat_3dgs(repo_dir)

    selected_frame_roots = [frame_root for _, frame_root in selected_timesteps]
    if selected_timesteps_need_sparse_normalization(selected_frame_roots):
        run_sparse_normalization(settings_path=settings_path, repo_root=repo_root, dry_run=dry_run)

    print(f"Wheat-3DGS repo: {repo_dir}")
    if patched_files:
        print(f"Patched Wheat-3DGS files: {patched_files}")
    else:
        print("Wheat-3DGS patches already applied")

    print("Using colleague-compatible masked RGB training.")

    env = build_environment(repo_dir)
    save_iterations = select_iterations(settings.num_iterations, DEFAULT_SAVE_ITERATIONS)
    test_iterations = select_iterations(settings.num_iterations, DEFAULT_TEST_ITERATIONS)

    for _, frame_root in selected_timesteps:
        masks_dir = ensure_prepared_timestep(frame_root)
        model_dir = build_model_dir(frame_root, settings)
        if model_dir.exists() and overwrite:
            shutil.rmtree(model_dir)

        print(f"\nProcessing {frame_root.name}")
        print(f"Active masks: {masks_dir}")
        print(f"Model output: {model_dir}")

        verify_masked_scene(
            repo_dir=repo_dir,
            frame_root=frame_root,
            model_dir=model_dir,
            resolution_decrease_factor=settings.resolution_decrease_factor,
            env=env,
            dry_run=dry_run,
        )

        train_command = [
            sys.executable,
            str(repo_dir / "train_vanilla_3dgs.py"),
            "-s",
            str(frame_root),
            "-m",
            str(model_dir),
            "--resolution",
            str(settings.resolution_decrease_factor),
            "--iterations",
            str(settings.num_iterations),
            "--save_iterations",
            *[str(value) for value in save_iterations],
            "--test_iterations",
            *[str(value) for value in test_iterations],
        ]
        render_command = [
            sys.executable,
            str(repo_dir / "render.py"),
            "-s",
            str(frame_root),
            "-m",
            str(model_dir),
            "--iteration",
            str(settings.num_iterations),
            "--resolution",
            str(settings.resolution_decrease_factor),
        ]

        run_command(train_command, repo_dir, env, dry_run)
        run_command(render_command, repo_dir, env, dry_run)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run masked Wheat-3DGS reconstructions with the colleague-compatible 16/6 split."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "settings_pipeline.txt",
        help="Path to the pipeline settings file.",
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external" / "Wheat-3DGS",
        help="Path to the cloned Wheat-3DGS repository.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without launching training.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing reconstruction output directory before rerunning a timestep.",
    )
    args = parser.parse_args()
    return run(
        settings_path=args.settings.resolve(),
        repo_dir=args.repo_dir.resolve(),
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
