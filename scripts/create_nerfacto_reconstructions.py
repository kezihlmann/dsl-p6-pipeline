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

from PIL import Image


TIMESTEP_PATTERN = re.compile(r"timestep_(\d+)$")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
DEFAULT_EXPORT_POINTS = 10000


@dataclass
class Settings:
    input_folder: str
    first_timestep: int
    last_timestep: int
    num_timesteps: int
    reconstruction_method: str
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
        reconstruction_method=str(values.get("reconstruction_method", "3dgs")),
        resolution_decrease_factor=int(values.get("resolution_decrease_factor", 1)),
        num_iterations=int(values.get("num_iterations", values.get("number_of_iteration", 30000))),
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


def ensure_sparse_zero(frame_root: Path) -> None:
    sparse_root = frame_root / "sparse"
    zero_dir = sparse_root / "0"
    if zero_dir.exists():
        return
    if not sparse_root.exists():
        raise FileNotFoundError(
            f"Missing COLMAP sparse model for {frame_root.name} at {frame_root}. "
            "Step 2 requires a provided sparse model before Nerfacto reconstruction can start. "
            "Expected either 'sparse/0' or a 'sparse' folder that can be normalized into 'sparse/0'."
        )

    zero_dir.mkdir(exist_ok=True)
    for path in list(sparse_root.iterdir()):
        if path.name == "0":
            continue
        shutil.move(str(path), str(zero_dir / path.name))


def find_mask(mask_dir: Path, image_path: Path) -> Path | None:
    stem = image_path.stem
    name = image_path.name
    camera_prefix = stem.split("_")[0]

    candidates = [
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}.jpg",
        mask_dir / f"{stem}.jpeg",
        mask_dir / name,
        mask_dir / f"{stem}_mask_ground_truth.png",
        mask_dir / f"{stem}_mask_sam3.png",
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{stem}_binary.png",
        mask_dir / f"mask_{stem}.png",
        mask_dir / f"binary_{stem}.png",
        mask_dir / f"{camera_prefix}.png",
        mask_dir / f"{camera_prefix}_mask.png",
        mask_dir / f"{camera_prefix}_binary.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(mask_dir.glob(f"*{stem}*"))
    if len(matches) == 1:
        return matches[0]

    matches = list(mask_dir.glob(f"*{camera_prefix}*"))
    if len(matches) == 1:
        return matches[0]

    return None


def resolve_mask_dir(frame_root: Path) -> Path:
    for candidate in (
        frame_root / "masks_binary_active",
        frame_root / "masks",
        frame_root / "masks_binary_gt",
        frame_root / "masks_binary",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No usable mask directory found for {frame_root}. Expected one of masks_binary_active, masks, masks_binary_gt, or masks_binary."
    )


def collect_images(images_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for extension in IMAGE_EXTENSIONS:
        image_paths.extend(images_dir.glob(f"*{extension}"))
    return sorted(set(image_paths))


def write_link_or_copy(source_path: Path, target_path: Path) -> None:
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    try:
        target_path.symlink_to(source_path.name)
    except OSError:
        shutil.copy2(source_path, target_path)


def prepare_downscaled_images(dataset_root: Path, factor: int) -> Path:
    source_dir = dataset_root / "images"
    target_dir = dataset_root / f"images_{factor}"
    target_dir.mkdir(parents=True, exist_ok=True)

    for png_path in sorted(source_dir.glob("*.png")):
        out_png = target_dir / png_path.name
        if not out_png.exists():
            image = Image.open(png_path).convert("RGBA")
            width, height = image.size
            downscaled = image.resize(
                (max(1, width // factor), max(1, height // factor)),
                Image.Resampling.LANCZOS,
            )
            downscaled.save(out_png)

        jpg_link = target_dir / f"{png_path.stem}.jpg"
        write_link_or_copy(out_png, jpg_link)

    return target_dir


def prepare_rgba_dataset(frame_root: Path, overwrite: bool) -> Path:
    ensure_sparse_zero(frame_root)

    images_dir = frame_root / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"No images folder found for {frame_root}")

    mask_dir = resolve_mask_dir(frame_root)
    dataset_root = frame_root / "nerfacto-rgba-dataset"
    if dataset_root.exists() and overwrite:
        shutil.rmtree(dataset_root)

    dataset_images = dataset_root / "images"
    dataset_sparse = dataset_root / "sparse"
    dataset_images.mkdir(parents=True, exist_ok=True)

    if dataset_sparse.exists():
        shutil.rmtree(dataset_sparse)
    shutil.copytree(frame_root / "sparse", dataset_sparse)

    image_paths = collect_images(images_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    missing_masks: list[str] = []
    for image_path in image_paths:
        mask_path = find_mask(mask_dir, image_path)
        if mask_path is None:
            missing_masks.append(image_path.name)
            continue

        out_path = dataset_images / f"{image_path.stem}.png"
        if out_path.exists() and not overwrite:
            continue

        rgb = Image.open(image_path).convert("RGB")
        alpha = Image.open(mask_path).convert("L")
        if alpha.size != rgb.size:
            alpha = alpha.resize(rgb.size, Image.Resampling.NEAREST)

        rgba = Image.merge("RGBA", (*rgb.split(), alpha))
        rgba.save(out_path)
        if image_path.suffix.lower() != ".png":
            alias_path = dataset_images / image_path.name
            write_link_or_copy(out_path, alias_path)

    if missing_masks:
        preview = ", ".join(missing_masks[:5])
        raise RuntimeError(
            f"Missing {len(missing_masks)} masks for {frame_root.name}. "
            f"First missing files: {preview}"
        )

    return dataset_root


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(
            f"Required executable '{name}' was not found on PATH. "
            "Activate the dedicated Nerfacto environment from environment-euler-nerfacto.yml before running Nerfacto reconstruction."
        )


def run_command(command: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print(f"Running: {' '.join(command)}")
    if dry_run:
        return
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    return env


def find_latest_config(experiment_root: Path) -> Path:
    config_paths = sorted(experiment_root.glob("nerfacto/*/config.yml"))
    if not config_paths:
        raise FileNotFoundError(f"No Nerfacto config.yml found in {experiment_root}")
    return config_paths[-1]


def write_render_config_override(config_path: Path, images_path_name: str) -> Path:
    override_path = config_path.with_name(f"{config_path.stem}_render_override.yml")
    lines = config_path.read_text(encoding="utf-8").splitlines()
    output_lines: list[str] = []
    image_replaced = False
    downscale_replaced = False
    skip_path_item = False

    for line in lines:
        stripped = line.strip()

        if skip_path_item:
            if stripped.startswith("- "):
                continue
            skip_path_item = False

        if stripped.startswith("images_path:"):
            indent = line[: len(line) - len(line.lstrip())]
            output_lines.append(f"{indent}images_path: !!python/object/apply:pathlib.PosixPath")
            output_lines.append(f"{indent}- {images_path_name}")
            image_replaced = True
            skip_path_item = True
            continue

        if stripped.startswith("downscale_factor:") and not downscale_replaced:
            indent = line[: len(line) - len(line.lstrip())]
            output_lines.append(f"{indent}downscale_factor: 1")
            downscale_replaced = True
            continue

        output_lines.append(line)

    if not image_replaced:
        raise ValueError(f"Failed to patch images_path in {config_path}")
    if not downscale_replaced:
        raise ValueError(f"Failed to patch downscale_factor in {config_path}")

    override_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    return override_path


def build_experiment_name(frame_root: Path, settings: Settings) -> str:
    resolution_label = "fullres" if settings.resolution_decrease_factor == 1 else f"down{settings.resolution_decrease_factor}"
    return f"nerfacto_rgba_{frame_root.name}_{resolution_label}_{settings.num_iterations}"


def run(settings_path: Path, dry_run: bool, overwrite: bool) -> int:
    settings = parse_settings(settings_path)
    repo_root = settings_path.parent
    input_root = (repo_root / settings.input_folder).resolve()

    if settings.reconstruction_method != "nerfacto":
        raise ValueError(
            f"This script only handles reconstruction_method='nerfacto', got {settings.reconstruction_method!r}."
        )
    if not input_root.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")

    require_executable("ns-train")
    require_executable("ns-render")
    require_executable("ns-export")

    env = build_environment()
    selected_timesteps = choose_timesteps(find_available_timesteps(input_root), settings)

    print("Using Nerfacto RGBA reconstruction based on the reference nerfstudio_project workflow.")

    for _, frame_root in selected_timesteps:
        dataset_root = prepare_rgba_dataset(frame_root, overwrite=overwrite)
        images_path_name = "images"
        train_downscale_factor = 1
        resolution_mode_message = (
            "Following the reference nerfstudio_project workflow: "
            "Nerfacto trains from images/ with --downscale-factor 1. "
            f"resolution_decrease_factor={settings.resolution_decrease_factor} is only applied later during rendering"
            " via a prepared downscaled RGBA image folder."
        )
        recon_root = frame_root / "nerfacto-reconstructions"
        experiment_name = build_experiment_name(frame_root, settings)
        experiment_root = recon_root / experiment_name
        render_root = experiment_root / f"renders_test_down{settings.resolution_decrease_factor}"
        export_root = experiment_root / f"pointcloud_{DEFAULT_EXPORT_POINTS}"

        if overwrite and experiment_root.exists():
            shutil.rmtree(experiment_root)

        print(f"\nProcessing {frame_root.name}")
        print(f"Prepared RGBA dataset: {dataset_root}")
        print(f"Images path for Nerfacto: {dataset_root / images_path_name}")
        print(resolution_mode_message)
        print(f"Experiment root: {experiment_root}")
        print(f"Target Nerfacto iterations: {settings.num_iterations}")

        train_command = [
            "ns-train",
            "nerfacto",
            "--output-dir",
            str(recon_root),
            "--experiment-name",
            experiment_name,
            "--max-num-iterations",
            str(settings.num_iterations),
            "--vis",
            "tensorboard",
            "--steps-per-save",
            "5000",
            "--steps-per-eval-image",
            "1000000",
            "--steps-per-eval-all-images",
            "1000000",
            "--pipeline.model.background-color",
            "random",
            "--pipeline.datamanager.train-num-rays-per-batch",
            "2048",
            "--pipeline.datamanager.eval-num-rays-per-batch",
            "2048",
            "colmap",
            "--data",
            str(dataset_root),
            "--images-path",
            images_path_name,
            "--colmap-path",
            "sparse/0",
            "--downscale-factor",
            str(train_downscale_factor),
        ]
        run_command(train_command, repo_root, env, dry_run)

        config_path = experiment_root / "_dry_run_config.yml" if dry_run else find_latest_config(experiment_root)
        render_config_path = config_path
        if settings.resolution_decrease_factor > 1:
            prepare_downscaled_images(dataset_root, settings.resolution_decrease_factor)
            if not dry_run:
                render_config_path = write_render_config_override(
                    config_path=config_path,
                    images_path_name=f"images_{settings.resolution_decrease_factor}",
                )
        render_command = [
            "ns-render",
            "dataset",
            "--load-config",
            str(render_config_path),
            "--output-path",
            str(render_root),
            "--split",
            "test",
            "--rendered-output-names",
            "rgb",
            "accumulation",
            "depth",
            "--image-format",
            "png",
            "--eval-num-rays-per-chunk",
            "4096",
            "--downscale-factor",
            "1",
        ]
        export_command = [
            "ns-export",
            "pointcloud",
            "--load-config",
            str(config_path),
            "--output-dir",
            str(export_root),
            "--num-points",
            str(DEFAULT_EXPORT_POINTS),
            "--num-rays-per-batch",
            "256",
            "--remove-outliers",
            "False",
            "--normal-method",
            "open3d",
            "--reorient-normals",
            "False",
        ]

        run_command(render_command, repo_root, env, dry_run)
        run_command(export_command, repo_root, env, dry_run)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Nerfacto RGBA reconstructions from masked timestep folders with provided COLMAP sparse models."
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "settings_pipeline.txt",
        help="Path to the pipeline settings file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without launching Nerfacto training, rendering, or export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing Nerfacto outputs and the prepared RGBA dataset before rerunning a timestep.",
    )
    args = parser.parse_args()
    return run(settings_path=args.settings.resolve(), dry_run=args.dry_run, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
