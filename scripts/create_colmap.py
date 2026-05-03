from __future__ import annotations

import argparse
import ast
import re
import shutil
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


def ensure_sparse_zero(frame_root: Path) -> None:
	sparse_root = frame_root / "sparse"
	if not sparse_root.exists():
		raise FileNotFoundError(f"No sparse directory found in {frame_root}")

	zero_dir = sparse_root / "0"
	if zero_dir.exists():
		return

	zero_dir.mkdir(exist_ok=True)
	for path in list(sparse_root.iterdir()):
		if path.name == "0":
			continue
		shutil.move(str(path), str(zero_dir / path.name))


def prepare_timestep(frame_root: Path) -> dict[str, str]:
	ensure_sparse_zero(frame_root)
	return {
		"timestep": frame_root.name,
	}


def run(settings_path: Path) -> int:
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
		result = prepare_timestep(timestep_dir)
		print(f"Prepared {result['timestep']}: ensured sparse/0")

	return 0


def main() -> int:
	parser = argparse.ArgumentParser(description="Prepare timestep folders for Wheat-3DGS reconstruction.")
	parser.add_argument(
		"--settings",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "settings_pipeline.txt",
		help="Path to the pipeline settings file.",
	)
	args = parser.parse_args()
	return run(args.settings.resolve())


if __name__ == "__main__":
	raise SystemExit(main())
