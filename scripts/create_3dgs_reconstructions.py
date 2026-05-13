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
DEFAULT_TEST_VIEW_COUNT = 5


@dataclass
class Settings:
	input_folder: str
	output_folder: str
	first_timestep: int
	last_timestep: int
	number_of_timesteps: int
	loss: str
	resolution_decrease_factor: int
	number_of_iteration: int


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
		loss=str(values.get("loss", "silhouette")),
		resolution_decrease_factor=int(values.get("resolution_decrease_factor", 2)),
		number_of_iteration=int(values.get("number_of_iteration", 15000)),
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


def ensure_contains_once(text: str, needle: str, replacement: str) -> str:
	if needle in text:
		return text
	return replacement


def patch_dataset_readers(repo_dir: Path) -> bool:
	path = repo_dir / "scene" / "dataset_readers.py"
	text = path.read_text(encoding="utf-8")
	original = text

	if "alpha_mask_path: str | None" not in text:
		text = text.replace(
			"    mask_paths: List[str]\n",
			"    mask_paths: List[str]\n    alpha_mask_path: str | None\n",
		)

	if 'alpha_mask_candidates = [' not in text:
		text = text.replace(
			"        image_path = os.path.join(images_folder, os.path.basename(extr.name))\n"
			"        image_name = os.path.basename(image_path).split(\".\")[0]\n"
			"        image = Image.open(image_path)\n",
			"        image_path = os.path.join(images_folder, os.path.basename(extr.name))\n"
			"        image_name = os.path.basename(image_path).split(\".\")[0]\n"
			"        image = Image.open(image_path)\n"
			"\n"
			"        alpha_mask_candidates = [\n"
			"            os.path.join(os.path.dirname(images_folder), \"masks\", image_name + \"_mask_sam3.png\"),\n"
			"        ]\n"
			"        alpha_mask_path = next(\n"
			"            (os.path.normpath(candidate) for candidate in alpha_mask_candidates if os.path.exists(candidate)),\n"
			"            None,\n"
			"        )\n",
		)

	if "alpha_mask_path=alpha_mask_path" not in text:
		text = text.replace(
			"        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,\n"
			"                              image_path=image_path, image_name=image_name, width=width, height=height,\n"
			"                              bbox_path=bbox_path, mask_paths=mask_paths)\n",
			"        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,\n"
			"                              image_path=image_path, image_name=image_name, width=width, height=height,\n"
			"                              bbox_path=bbox_path, mask_paths=mask_paths, alpha_mask_path=alpha_mask_path)\n",
		)

	if 'test_count = int(os.getenv("WHEAT3DGS_TEST_COUNT", "5"))' not in text:
		pattern = re.compile(
			r"    if eval:\n(?:.*\n)*?    else:\n        train_cam_infos = cam_infos\n        test_cam_infos = \[\]\n",
			re.MULTILINE,
		)
		replacement = (
			"    if eval:\n"
			"        requested_test_count = int(os.getenv(\"WHEAT3DGS_TEST_COUNT\", \"5\"))\n"
			"        if len(cam_infos) <= 1:\n"
			"            test_count = 0\n"
			"        else:\n"
			"            test_count = max(1, min(requested_test_count, len(cam_infos) - 1))\n"
			"        train_cam_infos = cam_infos[:-test_count] if test_count else cam_infos\n"
			"        test_cam_infos = cam_infos[-test_count:] if test_count else []\n"
			"        print(f\"Train Cam list with {len(train_cam_infos)} cams: {[cam.image_name for cam in train_cam_infos]}\")\n"
			"        print(f\"Test Cam list with {len(test_cam_infos)} cams: {[cam.image_name for cam in test_cam_infos]}\")\n"
			"    else:\n"
			"        train_cam_infos = cam_infos\n"
			"        test_cam_infos = []\n"
		)
		text, count = pattern.subn(replacement, text, count=1)
		if count != 1:
			raise ValueError("Failed to patch deterministic eval split in dataset_readers.py")

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

	old_block = (
		"    gt_image = resized_image_rgb[:3, ...]\n"
		"    loaded_mask = None\n\n"
		"    if resized_image_rgb.shape[1] == 4:\n"
		"        loaded_mask = resized_image_rgb[3:4, ...]\n\n"
		"    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, \n"
		"                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, \n"
		"                  image=gt_image, gt_alpha_mask=loaded_mask,\n"
		"                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,\n"
		"                  bbox_path=cam_info.bbox_path, mask_paths=cam_info.mask_paths, resolution=resolution, resolution_scale=scale)\n"
	)
	new_block = (
		"    gt_image = resized_image_rgb[:3, ...]\n"
		"    loaded_mask = None\n\n"
		"    if getattr(cam_info, \"alpha_mask_path\", None) is not None:\n"
		"        alpha_pil = Image.open(cam_info.alpha_mask_path).convert(\"L\")\n"
		"        alpha_resized = alpha_pil.resize(resolution, Image.NEAREST)\n"
		"        alpha_np = np.array(alpha_resized)\n"
		"        alpha_np = (alpha_np > 128).astype(np.float32)\n"
		"        loaded_mask = torch.from_numpy(alpha_np).unsqueeze(0)\n"
		"    elif resized_image_rgb.shape[1] == 4:\n"
		"        loaded_mask = resized_image_rgb[3:4, ...]\n\n"
		"    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, \n"
		"                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, \n"
		"                  image=gt_image, gt_alpha_mask=loaded_mask,\n"
		"                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,\n"
		"                  bbox_path=cam_info.bbox_path, mask_paths=cam_info.mask_paths, resolution=resolution, resolution_scale=scale)\n"
	)
	if 'getattr(cam_info, "alpha_mask_path", None)' not in text:
		if old_block not in text:
			raise ValueError("Failed to find loadCam block in camera_utils.py")
		text = text.replace(old_block, new_block)

	if text != original:
		path.write_text(text, encoding="utf-8")
		return True
	return False


def patch_cameras(repo_dir: Path) -> bool:
	path = repo_dir / "scene" / "cameras.py"
	text = path.read_text(encoding="utf-8")
	original = text

	old_block = (
		"        if gt_alpha_mask is not None:\n"
		"            self.original_image *= gt_alpha_mask.to(self.data_device)\n"
		"        else:\n"
		"            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)\n"
	)
	new_block = (
		"        if gt_alpha_mask is not None:\n"
		"            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)\n"
		"        else:\n"
		"            self.gt_alpha_mask = torch.ones((1, self.image_height, self.image_width), device=self.data_device)\n"
		"\n"
		"        self.original_image *= self.gt_alpha_mask\n"
	)
	if "self.gt_alpha_mask" not in text:
		if old_block not in text:
			raise ValueError("Failed to find alpha-mask block in cameras.py")
		text = text.replace(old_block, new_block)

	if text != original:
		path.write_text(text, encoding="utf-8")
		return True
	return False


def patch_train_vanilla_3dgs(repo_dir: Path) -> bool:
	path = repo_dir / "train_vanilla_3dgs.py"
	text = path.read_text(encoding="utf-8")
	original = text

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


def patch_wheatgs_helper(repo_dir: Path) -> bool:
	path = repo_dir / "utils" / "wheatgs_helper.py"
	text = path.read_text(encoding="utf-8")
	original = text

	old_import = "from shapely.geometry import Polygon\n"
	new_import = (
		"try:\n"
		"    from shapely.geometry import Polygon\n"
		"except ImportError:\n"
		"    Polygon = None\n"
	)
	if "Polygon = None" not in text:
		if old_import not in text:
			raise ValueError("Failed to find shapely import in wheatgs_helper.py")
		text = text.replace(old_import, new_import, 1)

	old_function = (
		"def polygon_from_points(points):\n"
		"    # Ensure points are in a proper order (if needed)\n"
		"    # For rectangles, points are typically in order (e.g. clockwise)\n"
		"    return Polygon(points)\n"
	)
	new_function = (
		"def polygon_from_points(points):\n"
		"    # Ensure points are in a proper order (if needed)\n"
		"    # For rectangles, points are typically in order (e.g. clockwise)\n"
		"    if Polygon is None:\n"
		"        raise ImportError(\"shapely is required for polygon matching helpers\")\n"
		"    return Polygon(points)\n"
	)
	if 'raise ImportError("shapely is required for polygon matching helpers")' not in text:
		if old_function not in text:
			raise ValueError("Failed to find polygon_from_points in wheatgs_helper.py")
		text = text.replace(old_function, new_function, 1)

	if text != original:
		path.write_text(text, encoding="utf-8")
		return True
	return False


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
	if patch_wheatgs_helper(repo_dir):
		changed.append("utils/wheatgs_helper.py")
	return changed


def ensure_prepared_timestep(frame_root: Path) -> None:
	if not (frame_root / "sparse" / "0").exists():
		raise FileNotFoundError(
			f"Prepared sparse/0 directory missing for {frame_root}. Run create_colmap.py first."
		)
	if not (frame_root / "masks").exists():
		raise FileNotFoundError(
			f"Mask directory missing for {frame_root}. Run create_sam3_masks.py first."
		)


def build_model_dir(frame_root: Path, settings: Settings) -> Path:
	label = f"wheat3dgs_{settings.loss}_res{settings.resolution_decrease_factor}_it{settings.number_of_iteration}"
	return frame_root / label


def build_environment(repo_dir: Path, test_view_count: int) -> dict[str, str]:
	env = os.environ.copy()
	python_path_entries = [
		str(repo_dir),
	]
	existing_pythonpath = env.get("PYTHONPATH")
	if existing_pythonpath:
		python_path_entries.append(existing_pythonpath)
	env["PYTHONPATH"] = os.pathsep.join(python_path_entries)
	env["WANDB_MODE"] = "disabled"
	env["WANDB_DISABLED"] = "true"
	env["WHEAT3DGS_TEST_COUNT"] = str(test_view_count)
	return env


def run_command(command: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> None:
	print(f"Running: {' '.join(command)}")
	if dry_run:
		return
	subprocess.run(command, cwd=cwd, env=env, check=True)


def run(settings_path: Path, repo_dir: Path, dry_run: bool, overwrite: bool, test_view_count: int) -> int:
	settings = parse_settings(settings_path)
	repo_root = settings_path.parent
	input_root = (repo_root / settings.input_folder).resolve()

	if not input_root.exists():
		raise FileNotFoundError(f"Input folder does not exist: {input_root}")

	available_timesteps = find_available_timesteps(input_root)
	selected_timesteps = choose_timesteps(available_timesteps, settings)
	patched_files = patch_wheat_3dgs(repo_dir)

	print(f"Wheat-3DGS repo: {repo_dir}")
	if patched_files:
		print(f"Patched Wheat-3DGS files: {patched_files}")
	else:
		print("Wheat-3DGS patches already applied")

	env = build_environment(repo_dir, test_view_count)

	for _, frame_root in selected_timesteps:
		ensure_prepared_timestep(frame_root)
		model_dir = build_model_dir(frame_root, settings)
		if model_dir.exists() and overwrite:
			shutil.rmtree(model_dir)

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
			str(settings.number_of_iteration),
		]
		render_command = [
			sys.executable,
			str(repo_dir / "render.py"),
			"-s",
			str(frame_root),
			"-m",
			str(model_dir),
			"--iteration",
			str(settings.number_of_iteration),
		]

		print(f"\nProcessing {frame_root.name}")
		run_command(train_command, repo_dir, env, dry_run)
		run_command(render_command, repo_dir, env, dry_run)

	return 0


def main() -> int:
	parser = argparse.ArgumentParser(description="Run Wheat-3DGS reconstructions for selected timestep folders.")
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
		help="Remove an existing model directory before running reconstruction.",
	)
	parser.add_argument(
		"--test-view-count",
		type=int,
		default=DEFAULT_TEST_VIEW_COUNT,
		help="Number of trailing camera views to reserve for evaluation.",
	)
	args = parser.parse_args()
	return run(
		settings_path=args.settings.resolve(),
		repo_dir=args.repo_dir.resolve(),
		dry_run=args.dry_run,
		overwrite=args.overwrite,
		test_view_count=args.test_view_count,
	)


if __name__ == "__main__":
	raise SystemExit(main())
