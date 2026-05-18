from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path


TIMESTEP_PATTERN = re.compile(r"timestep_(\d+)$")


@dataclass
class Settings:
	input_folder: str
	output_folder: str
	first_timestep: int
	last_timestep: int
	number_of_timesteps: int


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
		output_folder=str(values["output_folder"]),
		first_timestep=int(values["first_timestep"]),
		last_timestep=int(values["last_timestep"]),
		number_of_timesteps=int(values["number_of_timesteps"]),
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

	requested = min(settings.number_of_timesteps, len(filtered))
	if requested <= 0:
		raise ValueError("number_of_timesteps must be at least 1.")
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


def resolve_colmap_executable() -> str:
	colmap_executable = os.environ.get("COLMAP_EXECUTABLE", "colmap").strip()
	return colmap_executable or "colmap"


def run_command(command: list[str], cwd: Path) -> None:
	print(f"Running: {' '.join(command)}")
	try:
		subprocess.run(command, cwd=cwd, check=True)
	except FileNotFoundError as exc:
		if command and command[0] == resolve_colmap_executable():
			raise FileNotFoundError(
				"COLMAP executable not found. Install COLMAP in the environment or set "
				"COLMAP_EXECUTABLE to the full executable path before running create_colmap.py."
			) from exc
		raise


def ensure_sparse_zero(frame_root: Path) -> bool:
	sparse_root = frame_root / "sparse"
	if not sparse_root.exists():
		return False

	zero_dir = sparse_root / "0"
	if zero_dir.exists():
		return True

	zero_dir.mkdir(exist_ok=True)
	for path in list(sparse_root.iterdir()):
		if path.name == "0":
			continue
		shutil.move(str(path), str(zero_dir / path.name))
	return True


def reconstruct_sparse_model(frame_root: Path, use_gpu: bool) -> None:
	image_dir = frame_root / "images"
	if not image_dir.exists():
		raise FileNotFoundError(f"No images directory found in {frame_root}")

	colmap_executable = resolve_colmap_executable()
	colmap_root = frame_root / "colmap"
	distorted_root = colmap_root / "distorted"
	sparse_root = frame_root / "sparse"
	database_path = distorted_root / "database.db"
	mapping_root = distorted_root / "sparse"

	if colmap_root.exists():
		shutil.rmtree(colmap_root)
	if sparse_root.exists():
		shutil.rmtree(sparse_root)

	mapping_root.mkdir(parents=True, exist_ok=True)
	sift_gpu = "1" if use_gpu else "0"

	run_command(
		[
			colmap_executable,
			"feature_extractor",
			"--database_path",
			str(database_path),
			"--image_path",
			str(image_dir),
			"--ImageReader.single_camera",
			"1",
			"--ImageReader.camera_model",
			"PINHOLE",
			"--SiftExtraction.use_gpu",
			sift_gpu,
		],
		frame_root,
	)

	run_command(
		[
			colmap_executable,
			"exhaustive_matcher",
			"--database_path",
			str(database_path),
			"--SiftMatching.use_gpu",
			sift_gpu,
		],
		frame_root,
	)

	run_command(
		[
			colmap_executable,
			"mapper",
			"--database_path",
			str(database_path),
			"--image_path",
			str(image_dir),
			"--output_path",
			str(mapping_root),
			"--Mapper.ba_global_function_tolerance=0.000001",
		],
		frame_root,
	)

	if not ensure_sparse_zero(frame_root):
		raise FileNotFoundError(f"COLMAP reconstruction did not produce a sparse model for {frame_root}")


def prepare_timestep(frame_root: Path, use_gpu: bool) -> dict[str, str]:
	if ensure_sparse_zero(frame_root):
		return {
			"timestep": frame_root.name,
			"status": "reused existing sparse/0",
		}

	reconstruct_sparse_model(frame_root, use_gpu=use_gpu)
	return {
		"timestep": frame_root.name,
		"status": "created sparse/0 with COLMAP",
	}


def run(settings_path: Path, use_gpu: bool) -> int:
	settings = parse_settings(settings_path)
	repo_root = settings_path.parent
	input_root = (repo_root / settings.input_folder).resolve()

	if not input_root.exists():
		raise FileNotFoundError(f"Input folder does not exist: {input_root}")

	available_timesteps = find_available_timesteps(input_root)
	selected_timesteps = choose_timesteps(available_timesteps, settings)

	print(f"Input root: {input_root}")
	print(f"Selected timesteps: {[path.name for _, path in selected_timesteps]}")

	for _, timestep_dir in selected_timesteps:
		result = prepare_timestep(timestep_dir, use_gpu=use_gpu)
		print(f"Prepared {result['timestep']}: {result['status']}")

	return 0


def main() -> int:
	parser = argparse.ArgumentParser(description="Create or normalize COLMAP sparse models for selected timestep folders.")
	parser.add_argument(
		"--settings",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "settings_pipeline.txt",
		help="Path to the pipeline settings file.",
	)
	parser.add_argument(
		"--no-gpu",
		action="store_true",
		help="Disable GPU use for COLMAP SIFT extraction and matching.",
	)
	args = parser.parse_args()
	return run(args.settings.resolve(), use_gpu=not args.no_gpu)


if __name__ == "__main__":
	raise SystemExit(main())
