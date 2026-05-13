from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import get_token
from PIL import Image
from transformers import AutoTokenizer, Sam3ImageProcessor, Sam3Model, Sam3Processor


EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TIMESTEP_PATTERN = re.compile(r"timestep_(\d+)$")


@dataclass
class Settings:
	input_folder: str
	output_folder: str
	first_timestep: int
	last_timestep: int
	number_of_timesteps: int
	replace_existing_masks: bool
	sam_3_prompt: str
	threshold: float = 0.5
	mask_threshold: float = 0.5
	crop_padding: int = 160
	model_name: str = "facebook/sam3"


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
		replace_existing_masks=bool(values["replace_existing_masks"]),
		sam_3_prompt=str(values["sam_3_prompt"]),
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

	indices = np.linspace(0, len(filtered) - 1, num=requested)
	selected_indices = sorted({int(round(value)) for value in indices})
	while len(selected_indices) < requested:
		for index in range(len(filtered)):
			if index not in selected_indices:
				selected_indices.append(index)
			if len(selected_indices) == requested:
				break
	return [filtered[index] for index in sorted(selected_indices)]


def resolve_token() -> str | None:
	return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or get_token()


def load_model_and_processor(model_name: str, token: str | None, device: str) -> tuple[Sam3Model, Sam3Processor]:
	load_kwargs = {"token": token} if token else {}
	model = Sam3Model.from_pretrained(model_name, **load_kwargs).to(device)
	try:
		processor = Sam3Processor.from_pretrained(model_name, **load_kwargs)
	except OSError:
		# Some SAM3 Hub snapshots ship only processor_config.json, while the current
		# AutoImageProcessor path still looks for preprocessor_config.json first.
		tokenizer = AutoTokenizer.from_pretrained(model_name, **load_kwargs)
		image_processor = Sam3ImageProcessor()
		processor = Sam3Processor(image_processor=image_processor, tokenizer=tokenizer)
	return model, processor


def run_sam_text(
	image: Image.Image,
	text_prompt: str,
	processor: Sam3Processor,
	model: Sam3Model,
	device: str,
	threshold: float,
	mask_threshold: float,
):
	inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
	with torch.no_grad():
		outputs = model(**inputs)
	return processor.post_process_instance_segmentation(
		outputs,
		threshold=threshold,
		mask_threshold=mask_threshold,
		target_sizes=inputs["original_sizes"].tolist(),
	)[0]


def select_best_result(results) -> dict[str, object] | None:
	if len(results["scores"]) == 0:
		return None
	if isinstance(results["scores"], torch.Tensor):
		best_index = int(torch.argmax(results["scores"]).item())
	else:
		best_index = int(np.argmax(results["scores"]))
	return {
		"box": results["boxes"][best_index],
		"mask": results["masks"][best_index],
	}


def to_numpy_mask(mask) -> np.ndarray:
	if isinstance(mask, torch.Tensor):
		return mask.detach().cpu().numpy().astype(np.uint8)
	return np.asarray(mask).astype(np.uint8)


def to_numpy_box(box) -> np.ndarray:
	if isinstance(box, torch.Tensor):
		return box.detach().cpu().numpy().astype(int)
	return np.asarray(box).astype(int)


def expand_box(box: np.ndarray, width: int, height: int, pad: int) -> list[int]:
	x1, y1, x2, y2 = box.tolist()
	return [
		max(0, x1 - pad),
		max(0, y1 - pad),
		min(width, x2 + pad),
		min(height, y2 + pad),
	]


def process_image(
	image_path: Path,
	output_dir: Path,
	settings: Settings,
	processor: Sam3Processor,
	model: Sam3Model,
	device: str,
) -> bool:
	try:
		image = Image.open(image_path).convert("RGB")
	except Exception as exc:
		print(f"Failed to load {image_path}: {exc}", file=sys.stderr)
		return False

	width, height = image.size
	results = run_sam_text(
		image=image,
		text_prompt=settings.sam_3_prompt,
		processor=processor,
		model=model,
		device=device,
		threshold=settings.threshold,
		mask_threshold=settings.mask_threshold,
	)
	best = select_best_result(results)
	if best is None:
		print(f"No detection for {image_path.name}")
		return False

	box = to_numpy_box(best["box"])
	mask_full = to_numpy_mask(best["mask"])
	x1, y1, x2, y2 = expand_box(box, width, height, settings.crop_padding)
	crop = image.crop((x1, y1, x2, y2))

	crop_results = run_sam_text(
		image=crop,
		text_prompt=settings.sam_3_prompt,
		processor=processor,
		model=model,
		device=device,
		threshold=settings.threshold,
		mask_threshold=settings.mask_threshold,
	)
	crop_best = select_best_result(crop_results)
	if crop_best is None:
		final_mask = mask_full
	else:
		crop_mask = to_numpy_mask(crop_best["mask"])
		final_mask = np.zeros((height, width), dtype=np.uint8)
		final_mask[y1:y2, x1:x2] = crop_mask

	output_path = output_dir / f"{image_path.stem}_mask_sam3.png"
	Image.fromarray((final_mask > 0).astype(np.uint8) * 255).save(output_path)
	print(f"Saved {output_path}")
	return True


def run(settings_path: Path, device_override: str | None = None) -> int:
	settings = parse_settings(settings_path)
	repo_root = settings_path.parent
	input_root = (repo_root / settings.input_folder).resolve()
	device = device_override or "cuda"
	token = resolve_token()

	if not input_root.exists():
		raise FileNotFoundError(f"Input folder does not exist: {input_root}")
	if device != "cuda":
		raise ValueError("SAM3 mask generation only supports --device cuda.")
	if not torch.cuda.is_available():
		raise RuntimeError(
			"SAM3 mask generation requires a CUDA GPU, but torch.cuda.is_available() is False. "
			"Run this step on a GPU node and make sure the CUDA-enabled environment is activated."
		)

	print(f"Using device: {device}")
	print(f"Input root: {input_root}")
	print("Prepared masks will be written into each timestep folder under masks.")
	if token:
		print("Using Hugging Face token from environment.")
	else:
		print("No Hugging Face token in environment. Using cached/public access.")

	available_timesteps = find_available_timesteps(input_root)
	selected_timesteps = choose_timesteps(available_timesteps, settings)
	model, processor = load_model_and_processor(settings.model_name, token, device)

	total_processed = 0
	total_skipped = 0
	for timestep_value, timestep_input_dir in selected_timesteps:
		timestep_name = timestep_input_dir.name
		image_dir = timestep_input_dir / "images"
		output_dir = timestep_input_dir / "masks"
		if not image_dir.exists():
			print(f"Skipping {timestep_name}: no images folder")
			continue

		output_dir.mkdir(parents=True, exist_ok=True)
		images = [
			path for path in sorted(image_dir.iterdir()) if path.is_file() and path.suffix.lower() in EXTENSIONS
		]
		print(f"\nProcessing {timestep_name} ({timestep_value})")
		print(f"Images: {len(images)}")

		folder_processed = 0
		folder_skipped = 0
		for image_path in images:
			output_path = output_dir / f"{image_path.stem}_mask_sam3.png"
			if output_path.exists() and not settings.replace_existing_masks:
				folder_skipped += 1
				total_skipped += 1
				continue

			success = process_image(image_path, output_dir, settings, processor, model, device)
			if success:
				folder_processed += 1
				total_processed += 1

		print(f"Finished {timestep_name}: {folder_processed} new, {folder_skipped} skipped in {output_dir.name}")

	print(f"\nFinished all folders: {total_processed} new, {total_skipped} skipped")
	return 0


def main() -> int:
	parser = argparse.ArgumentParser(description="Generate SAM3 masks for selected timestep folders.")
	parser.add_argument(
		"--settings",
		type=Path,
		default=Path(__file__).resolve().parents[1] / "settings_pipeline.txt",
		help="Path to the pipeline settings file.",
	)
	parser.add_argument(
		"--device",
		choices=["cuda"],
		default=None,
		help="SAM3 requires CUDA and will only run on a GPU.",
	)
	args = parser.parse_args()
	return run(args.settings.resolve(), args.device)


if __name__ == "__main__":
	raise SystemExit(main())
