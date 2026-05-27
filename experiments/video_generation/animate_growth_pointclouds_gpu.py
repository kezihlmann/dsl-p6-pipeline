from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_AXIS_FILE = Path(__file__).resolve().parent / "stem_axis.txt"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "plant_growth_orbit.mp4"
DEFAULT_FPS = 24.0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_SECONDS_PER_TRANSITION = 2.0
DEFAULT_HOLD_SECONDS = 0.5
DEFAULT_START_AZIM = -60.0
DEFAULT_DEGREES_PER_TRANSITION = 120.0
DEFAULT_ELEVATION = 18.0
DEFAULT_ROLL = 0.0
DEFAULT_PADDING = 0.18
DEFAULT_VERTICAL_FOV = 35.0
DEFAULT_MIN_OPACITY = 0.02
DEFAULT_SIGMA_CUTOFF = 2.6
DEFAULT_SPLAT_SCALE = 1.0
DEFAULT_MAX_SPLAT_RADIUS = 90
CODEC = "mp4v"
TIMESTEP_PATTERN = re.compile(r"(\d+)(?!.*\d)")
SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class LoadedGaussianCloud:
    path: Path
    means: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray
    covariances: np.ndarray
    label: str
    stem_axis: tuple[np.ndarray, np.ndarray] | None


@dataclass(frozen=True)
class CameraPose:
    eye: np.ndarray
    rotation_world_to_camera: np.ndarray


@dataclass(frozen=True)
class PLYHeader:
    text: str
    data_offset: int


@dataclass(frozen=True)
class TorchGaussianCloud:
    means: "torch.Tensor"
    colors: "torch.Tensor"
    opacities: "torch.Tensor"
    covariances: "torch.Tensor"


@dataclass(frozen=True)
class FrameRenderTask:
    frame_index: int
    left_index: int
    right_index: int
    alpha: float
    pose: CameraPose


_FRAME_WORKER_CLOUDS: list[LoadedGaussianCloud] | None = None
_FRAME_WORKER_RENDERER: "GaussianSplatRenderer | None" = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render aligned plant Gaussian-splat PLY files into one orbiting growth video. "
            "Adjacent timesteps are cross-faded while the camera moves around the plant."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory that contains the timestep PLY files. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output MP4 path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--axis-file",
        type=Path,
        default=DEFAULT_AXIS_FILE,
        help=f"Stem axis text file with start= and end= coordinates. Default: {DEFAULT_AXIS_FILE}",
    )
    parser.add_argument(
        "--skip-axis-sync",
        action="store_true",
        help="Do not copy the axis-file coordinates into the PLY headers before rendering.",
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help=f"Video frame rate. Default: {DEFAULT_FPS}")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Output width. Default: {DEFAULT_WIDTH}")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Output height. Default: {DEFAULT_HEIGHT}")
    parser.add_argument(
        "--seconds-per-transition",
        type=float,
        default=DEFAULT_SECONDS_PER_TRANSITION,
        help=f"Seconds for each timestep-to-timestep blend. Default: {DEFAULT_SECONDS_PER_TRANSITION}",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=DEFAULT_HOLD_SECONDS,
        help=f"Seconds to hold the first and last timesteps. Default: {DEFAULT_HOLD_SECONDS}",
    )
    parser.add_argument(
        "--start-azim",
        type=float,
        default=DEFAULT_START_AZIM,
        help=f"Starting azimuth angle in degrees. Default: {DEFAULT_START_AZIM}",
    )
    parser.add_argument(
        "--degrees-per-transition",
        type=float,
        default=DEFAULT_DEGREES_PER_TRANSITION,
        help=f"Orbit degrees covered during one timestep transition. Default: {DEFAULT_DEGREES_PER_TRANSITION}",
    )
    parser.add_argument(
        "--elevation",
        type=float,
        default=DEFAULT_ELEVATION,
        help=f"Camera elevation in degrees. Default: {DEFAULT_ELEVATION}",
    )
    parser.add_argument(
        "--roll",
        type=float,
        default=DEFAULT_ROLL,
        help=f"Camera roll in degrees. Default: {DEFAULT_ROLL}",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=DEFAULT_PADDING,
        help=f"Padding around the global scene box. Default: {DEFAULT_PADDING}",
    )
    parser.add_argument(
        "--vertical-fov",
        type=float,
        default=DEFAULT_VERTICAL_FOV,
        help=f"Vertical field of view in degrees. Default: {DEFAULT_VERTICAL_FOV}",
    )
    parser.add_argument(
        "--min-opacity",
        type=float,
        default=DEFAULT_MIN_OPACITY,
        help=f"Discard splats below this activated opacity. Default: {DEFAULT_MIN_OPACITY}",
    )
    parser.add_argument(
        "--sigma-cutoff",
        type=float,
        default=DEFAULT_SIGMA_CUTOFF,
        help=f"Ellipse support radius in sigma units. Default: {DEFAULT_SIGMA_CUTOFF}",
    )
    parser.add_argument(
        "--splat-scale",
        type=float,
        default=DEFAULT_SPLAT_SCALE,
        help=f"Multiplier applied to Gaussian scales before rendering. Default: {DEFAULT_SPLAT_SCALE}",
    )
    parser.add_argument(
        "--max-splat-radius",
        type=int,
        default=DEFAULT_MAX_SPLAT_RADIUS,
        help=f"Maximum projected splat radius in pixels. Default: {DEFAULT_MAX_SPLAT_RADIUS}",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Rendering device. 'auto' uses CUDA when PyTorch with CUDA is available, otherwise CPU.",
    )
    parser.add_argument(
        "--frame-workers",
        type=int,
        default=1,
        help="Number of worker processes for frame rendering on CPU. Default: 1",
    )
    return parser.parse_args()


def resolve_render_device(requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"

    if requested_device == "cuda":
        if torch is None:
            raise RuntimeError("--device cuda requires PyTorch to be installed.")
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return "cuda"

    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_axis_file(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Axis file does not exist: {path}")

    values: dict[str, tuple[float, float, float]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid line in axis file: {raw_line}")

        key, raw_value = [part.strip() for part in line.split("=", 1)]
        parts = [part.strip() for part in raw_value.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Expected x,y,z for '{key}' in: {path}")
        values[key] = (float(parts[0]), float(parts[1]), float(parts[2]))

    if "start" not in values or "end" not in values:
        raise ValueError(f"Axis file must define both 'start=' and 'end=' lines: {path}")

    return values["start"], values["end"]


def format_stem_comment(label: str, point: tuple[float, float, float]) -> str:
    return f"comment {label} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n"


def update_ply_header_with_axis(path: Path, start: tuple[float, float, float], end: tuple[float, float, float]) -> None:
    data = path.read_bytes()
    marker = b"end_header\n"
    header_end = data.find(marker)
    if header_end < 0:
        raise ValueError(f"Could not find end_header in: {path}")

    header_text = data[: header_end + len(marker)].decode("ascii", errors="strict")
    payload = data[header_end + len(marker):]
    header_lines = header_text.splitlines(keepends=True)

    filtered_lines: list[str] = []
    for line in header_lines:
        stripped = line.strip()
        if stripped.startswith("comment stem_axis_start ") or stripped.startswith("comment stem_axis_end "):
            continue
        if stripped == "end_header":
            filtered_lines.append(format_stem_comment("stem_axis_start", start))
            filtered_lines.append(format_stem_comment("stem_axis_end", end))
        filtered_lines.append(line)

    path.write_bytes("".join(filtered_lines).encode("ascii") + payload)


def sync_axis_file_to_plys(axis_file: Path, data_dir: Path) -> None:
    start, end = parse_axis_file(axis_file)
    ply_paths = sorted(path for path in data_dir.iterdir() if path.suffix.lower() == ".ply")
    if not ply_paths:
        raise FileNotFoundError(f"No .ply files were found in: {data_dir}")
    for path in ply_paths:
        update_ply_header_with_axis(path, start, end)
        print(f"Updated stem axis in: {path}")


def print_progress(current: int, total: int, label: str) -> None:
    total = max(total, 1)
    ratio = min(max(current / total, 0.0), 1.0)
    if sys.stdout.isatty():
        width = 30
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\r{label}: [{bar}] {current}/{total} ({ratio * 100:5.1f}%)", end="", flush=True)
        if current >= total:
            print()
        return

    print(f"{label}: {current}/{total} ({ratio * 100:5.1f}%)", flush=True)


def timestep_label(path: Path) -> str:
    match = TIMESTEP_PATTERN.search(path.stem)
    return match.group(1) if match else path.stem


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sh_dc_to_rgb(features_dc: np.ndarray) -> np.ndarray:
    return np.clip(features_dc * SH_C0 + 0.5, 0.0, 1.0)


def quaternion_to_rotation(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    normalized = quaternions / np.clip(norms, 1e-12, None)

    r = normalized[:, 0]
    x = normalized[:, 1]
    y = normalized[:, 2]
    z = normalized[:, 3]

    rotations = np.empty((normalized.shape[0], 3, 3), dtype=np.float32)
    rotations[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    rotations[:, 0, 1] = 2.0 * (x * y - r * z)
    rotations[:, 0, 2] = 2.0 * (x * z + r * y)
    rotations[:, 1, 0] = 2.0 * (x * y + r * z)
    rotations[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    rotations[:, 1, 2] = 2.0 * (y * z - r * x)
    rotations[:, 2, 0] = 2.0 * (x * z - r * y)
    rotations[:, 2, 1] = 2.0 * (y * z + r * x)
    rotations[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return rotations


def read_ply_header(path: Path) -> PLYHeader:
    data = path.read_bytes()
    header_end = data.find(b"end_header\n")
    if header_end < 0:
        raise ValueError(f"Could not find end_header in: {path}")

    data_offset = header_end + len(b"end_header\n")
    header_text = data[:data_offset].decode("ascii", errors="strict")
    return PLYHeader(text=header_text, data_offset=data_offset)


def parse_stem_axis_from_header(header_text: str) -> tuple[np.ndarray, np.ndarray] | None:
    start: np.ndarray | None = None
    end: np.ndarray | None = None

    for line in header_text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5 or parts[0] != "comment":
            continue
        if parts[1] == "stem_axis_start":
            start = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float32)
        elif parts[1] == "stem_axis_end":
            end = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float32)

    if start is None or end is None:
        return None

    if np.linalg.norm(end - start) < 1e-8:
        return None

    return start, end


def read_binary_gaussian_ply(path: Path) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    header = read_ply_header(path)
    data = path.read_bytes()

    property_names = [line.split()[-1] for line in header.text.splitlines() if line.startswith("property ")]
    vertex_line = next((line for line in header.text.splitlines() if line.startswith("element vertex ")), None)
    if vertex_line is None:
        raise ValueError(f"Could not find vertex count in: {path}")

    vertex_count = int(vertex_line.split()[-1])
    dtype = np.dtype([(name, "<f4") for name in property_names])
    array = np.frombuffer(data[header.data_offset:], dtype=dtype, count=vertex_count)
    if len(array) != vertex_count:
        raise ValueError(f"Expected {vertex_count} vertices but read {len(array)} from: {path}")
    return array, parse_stem_axis_from_header(header.text)


def load_gaussian_cloud(path: Path, min_opacity: float, splat_scale: float) -> LoadedGaussianCloud:
    vertices, stem_axis = read_binary_gaussian_ply(path)

    means = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float32)
    colors = sh_dc_to_rgb(
        np.column_stack([vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"]]).astype(np.float32)
    )
    opacities = sigmoid(vertices["opacity"].astype(np.float32))
    scales = np.exp(
        np.column_stack([vertices["scale_0"], vertices["scale_1"], vertices["scale_2"]]).astype(np.float32)
    )
    scales *= max(splat_scale, 1e-6)
    rotations = quaternion_to_rotation(
        np.column_stack([vertices["rot_0"], vertices["rot_1"], vertices["rot_2"], vertices["rot_3"]]).astype(np.float32)
    )

    diagonal = np.zeros((len(vertices), 3, 3), dtype=np.float32)
    diagonal[:, 0, 0] = scales[:, 0] * scales[:, 0]
    diagonal[:, 1, 1] = scales[:, 1] * scales[:, 1]
    diagonal[:, 2, 2] = scales[:, 2] * scales[:, 2]
    covariances = rotations @ diagonal @ np.transpose(rotations, (0, 2, 1))

    valid = np.isfinite(means).all(axis=1)
    valid &= np.isfinite(colors).all(axis=1)
    valid &= np.isfinite(opacities)
    valid &= np.isfinite(covariances).all(axis=(1, 2))
    valid &= opacities >= min_opacity

    means = means[valid]
    colors = colors[valid]
    opacities = opacities[valid]
    covariances = covariances[valid]

    if means.size == 0:
        raise ValueError(f"No valid Gaussians left after filtering: {path}")

    return LoadedGaussianCloud(
        path=path,
        means=means,
        colors=colors,
        opacities=opacities,
        covariances=covariances,
        label=timestep_label(path),
        stem_axis=stem_axis,
    )


def discover_point_clouds(data_dir: Path, min_opacity: float, splat_scale: float) -> list[LoadedGaussianCloud]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    ply_paths = sorted(path for path in data_dir.iterdir() if path.suffix.lower() == ".ply")
    if not ply_paths:
        raise FileNotFoundError(f"No .ply files were found in: {data_dir}")

    clouds: list[LoadedGaussianCloud] = []
    for index, path in enumerate(ply_paths, start=1):
        clouds.append(load_gaussian_cloud(path, min_opacity=min_opacity, splat_scale=splat_scale))
        print_progress(index, len(ply_paths), "Loading Gaussian splats")
    return clouds


def global_scene_limits(clouds: list[LoadedGaussianCloud], padding_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    mins = np.min([cloud.means.min(axis=0) for cloud in clouds], axis=0)
    maxs = np.max([cloud.means.max(axis=0) for cloud in clouds], axis=0)
    center = (mins + maxs) / 2.0
    half_extent = np.max(maxs - mins) / 2.0
    padded_half_extent = half_extent * (1.0 + padding_fraction)
    lower = center - padded_half_extent
    upper = center + padded_half_extent
    return lower.astype(np.float32), upper.astype(np.float32)


def build_frame_schedule(cloud_count: int, transition_frames: int, hold_frames: int) -> list[tuple[int, int, float]]:
    if cloud_count == 1:
        return [(0, 0, 0.0)] * max(transition_frames + hold_frames * 2, 1)

    schedule: list[tuple[int, int, float]] = []
    schedule.extend((0, 0, 0.0) for _ in range(hold_frames))
    for index in range(cloud_count - 1):
        for step in range(transition_frames):
            alpha = 1.0 if transition_frames == 1 else step / (transition_frames - 1)
            schedule.append((index, index + 1, alpha))
    schedule.extend((cloud_count - 1, cloud_count - 1, 1.0) for _ in range(hold_frames))
    return schedule


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.clip(np.linalg.norm(vector), 1e-12, None)


def perpendicular_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = normalize(direction)
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(direction, reference))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    basis_a = normalize(np.cross(direction, reference))
    basis_b = normalize(np.cross(direction, basis_a))
    return basis_a.astype(np.float32), basis_b.astype(np.float32)


def look_at_pose(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray, roll_deg: float) -> CameraPose:
    forward = normalize(target - eye)
    right = normalize(np.cross(forward, up_hint))
    if np.linalg.norm(right) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = normalize(np.cross(forward, fallback))
    up = normalize(np.cross(right, forward))

    if roll_deg != 0.0:
        roll = math.radians(roll_deg)
        cos_roll = math.cos(roll)
        sin_roll = math.sin(roll)
        rotated_right = right * cos_roll + up * sin_roll
        rotated_up = -right * sin_roll + up * cos_roll
        right = rotated_right.astype(np.float32)
        up = rotated_up.astype(np.float32)

    rotation = np.stack([right, up, forward], axis=0)
    return CameraPose(eye=eye.astype(np.float32), rotation_world_to_camera=rotation.astype(np.float32))


def camera_pose(center: np.ndarray, distance: float, azimuth_deg: float, elevation_deg: float, roll_deg: float) -> CameraPose:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)

    eye = center + np.array(
        [
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        ],
        dtype=np.float32,
    )

    return look_at_pose(eye=eye, target=center, up_hint=np.array([0.0, 0.0, 1.0], dtype=np.float32), roll_deg=roll_deg)


def stem_orbit_pose(
    stem_start: np.ndarray,
    stem_end: np.ndarray,
    distance: float,
    azimuth_deg: float,
    roll_deg: float,
) -> CameraPose:
    axis_direction = normalize(stem_end - stem_start)
    radial_a, radial_b = perpendicular_basis(axis_direction)
    azimuth = math.radians(azimuth_deg)
    midpoint = (stem_start + stem_end) * 0.5

    radial = math.cos(azimuth) * radial_a + math.sin(azimuth) * radial_b
    eye = midpoint + distance * radial

    return look_at_pose(eye=eye, target=midpoint, up_hint=axis_direction, roll_deg=roll_deg)


def choose_stem_axis(clouds: list[LoadedGaussianCloud]) -> tuple[np.ndarray, np.ndarray] | None:
    for cloud in clouds:
        if cloud.stem_axis is not None:
            return cloud.stem_axis
    return None


def estimate_scene_stem_direction(clouds: list[LoadedGaussianCloud]) -> np.ndarray:
    points = np.vstack([cloud.means for cloud in clouds])
    z = points[:, 2]
    z_min = float(np.quantile(z, 0.02))
    z_max = float(np.quantile(z, 0.55))
    edges = np.linspace(z_min, z_max, 7)

    centers: list[np.ndarray] = []
    for lower_edge, upper_edge in zip(edges[:-1], edges[1:]):
        slab = points[(z >= lower_edge) & (z < upper_edge)]
        if len(slab) < 20:
            continue
        centers.append(np.array([np.median(slab[:, 0]), np.median(slab[:, 1]), np.mean(slab[:, 2])], dtype=np.float32))

    if len(centers) < 2:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)

    center_points = np.stack(centers, axis=0)
    centered = center_points - center_points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0].astype(np.float32)
    if direction[2] < 0.0:
        direction = -direction
    return normalize(direction)


def axis_midpoint_distance_to_box(axis: tuple[np.ndarray, np.ndarray], lower: np.ndarray, upper: np.ndarray) -> float:
    midpoint = (axis[0] + axis[1]) / 2.0
    clamped = np.minimum(np.maximum(midpoint, lower), upper)
    return float(np.linalg.norm(midpoint - clamped))


def resolve_stem_axis_for_scene(
    stem_axis: tuple[np.ndarray, np.ndarray] | None,
    lower: np.ndarray,
    upper: np.ndarray,
    estimated_direction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if stem_axis is None:
        return None

    best_axis = stem_axis
    best_score = float("inf")
    best_signs = (1.0, 1.0, 1.0)

    for signs in itertools.product([-1.0, 1.0], repeat=3):
        sign_vector = np.array(signs, dtype=np.float32)
        candidate = (stem_axis[0] * sign_vector, stem_axis[1] * sign_vector)
        midpoint_distance = axis_midpoint_distance_to_box(candidate, lower, upper)
        candidate_direction = normalize(candidate[1] - candidate[0])
        alignment_error = 1.0 - abs(float(np.dot(candidate_direction, estimated_direction)))
        score = midpoint_distance * 10.0 + alignment_error
        if score < best_score:
            best_score = score
            best_axis = candidate
            best_signs = signs

    if best_signs != (1.0, 1.0, 1.0):
        print(f"Using transformed stem axis with sign flips {best_signs} to match the plant coordinate system.")

    best_direction = normalize(best_axis[1] - best_axis[0])
    if float(np.dot(best_direction, estimated_direction)) < 0.0:
        best_axis = (best_axis[1], best_axis[0])
        print("Swapped stem axis endpoints so the camera up direction follows plant growth.")

    return best_axis


def add_overlay(frame: np.ndarray, text: str) -> np.ndarray:
    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    padding = 16
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(
        overlay,
        (padding, padding),
        (padding * 2 + text_width, padding * 2 + text_height + baseline),
        (255, 255, 255),
        thickness=-1,
    )
    cv2.putText(
        overlay,
        text,
        (padding, padding + text_height),
        font,
        font_scale,
        (40, 50, 40),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return overlay


def frame_label(left: LoadedGaussianCloud, right: LoadedGaussianCloud, alpha: float) -> str:
    if left.label == right.label or alpha <= 0.0:
        return f"timestep {left.label}"
    if alpha >= 1.0:
        return f"timestep {right.label}"
    return f"timestep {left.label} -> {right.label}"


def blend_frames(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return first
    if alpha >= 1.0:
        return second
    return cv2.addWeighted(first, 1.0 - alpha, second, alpha, 0.0).astype(np.uint8, copy=False)


def frame_pose(
    frame_index: int,
    total_frames: int,
    start_azim: float,
    total_rotation: float,
    stem_axis: tuple[np.ndarray, np.ndarray] | None,
    scene_center: np.ndarray,
    distance: float,
    elevation: float,
    roll: float,
) -> CameraPose:
    progress = 0.0 if total_frames == 1 else (frame_index - 1) / (total_frames - 1)
    azimuth = start_azim + total_rotation * progress
    if stem_axis is None:
        return camera_pose(scene_center, distance, azimuth, elevation, roll)
    return stem_orbit_pose(
        stem_start=stem_axis[0],
        stem_end=stem_axis[1],
        distance=distance,
        azimuth_deg=azimuth,
        roll_deg=roll,
    )


def build_render_tasks(
    schedule: list[tuple[int, int, float]],
    start_azim: float,
    total_rotation: float,
    stem_axis: tuple[np.ndarray, np.ndarray] | None,
    scene_center: np.ndarray,
    distance: float,
    elevation: float,
    roll: float,
) -> list[FrameRenderTask]:
    total_frames = len(schedule)
    tasks: list[FrameRenderTask] = []
    for frame_index, (left_index, right_index, alpha) in enumerate(schedule, start=1):
        tasks.append(
            FrameRenderTask(
                frame_index=frame_index,
                left_index=left_index,
                right_index=right_index,
                alpha=alpha,
                pose=frame_pose(
                    frame_index=frame_index,
                    total_frames=total_frames,
                    start_azim=start_azim,
                    total_rotation=total_rotation,
                    stem_axis=stem_axis,
                    scene_center=scene_center,
                    distance=distance,
                    elevation=elevation,
                    roll=roll,
                ),
            )
        )
    return tasks


def init_frame_worker(
    clouds: list[LoadedGaussianCloud],
    width: int,
    height: int,
    vertical_fov: float,
    sigma_cutoff: float,
    max_splat_radius: int,
    device: str,
) -> None:
    global _FRAME_WORKER_CLOUDS, _FRAME_WORKER_RENDERER
    _FRAME_WORKER_CLOUDS = clouds
    _FRAME_WORKER_RENDERER = GaussianSplatRenderer(
        width=width,
        height=height,
        vertical_fov_deg=vertical_fov,
        sigma_cutoff=sigma_cutoff,
        max_splat_radius=max_splat_radius,
        device=device,
    )


def render_frame_task(task: FrameRenderTask) -> tuple[int, np.ndarray]:
    if _FRAME_WORKER_CLOUDS is None or _FRAME_WORKER_RENDERER is None:
        raise RuntimeError("Frame worker was not initialized.")

    left_cloud = _FRAME_WORKER_CLOUDS[task.left_index]
    right_cloud = _FRAME_WORKER_CLOUDS[task.right_index]
    left_frame = _FRAME_WORKER_RENDERER.render(left_cloud, task.pose)
    if task.left_index == task.right_index or task.alpha <= 0.0:
        frame = left_frame
    else:
        right_frame = _FRAME_WORKER_RENDERER.render(right_cloud, task.pose)
        frame = blend_frames(left_frame, right_frame, task.alpha)

    frame = add_overlay(frame, frame_label(left_cloud, right_cloud, task.alpha))
    return task.frame_index, frame


class GaussianSplatRenderer:
    def __init__(
        self,
        width: int,
        height: int,
        vertical_fov_deg: float,
        sigma_cutoff: float,
        max_splat_radius: int,
        device: str,
    ) -> None:
        self.width = width
        self.height = height
        self.sigma_cutoff = float(sigma_cutoff)
        self.max_splat_radius = int(max_splat_radius)
        self.cx = width / 2.0
        self.cy = height / 2.0
        self.fy = (height / 2.0) / math.tan(math.radians(vertical_fov_deg) / 2.0)
        self.fx = self.fy
        self.device = device
        self._torch_cloud_cache: dict[Path, TorchGaussianCloud] = {}
        self._numpy_grid_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._torch_grid_cache: dict[tuple[int, int], tuple["torch.Tensor", "torch.Tensor"]] = {}

    def render(self, cloud: LoadedGaussianCloud, pose: CameraPose) -> np.ndarray:
        if self.device == "cuda":
            return self._render_torch(cloud, pose)
        return self._render_numpy(cloud, pose)

    def _numpy_local_grid(self, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        key = (height, width)
        cached = self._numpy_grid_cache.get(key)
        if cached is not None:
            return cached

        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float32),
            np.arange(width, dtype=np.float32),
            indexing="ij",
        )
        cached = (grid_x, grid_y)
        self._numpy_grid_cache[key] = cached
        return cached

    def _torch_local_grid(self, height: int, width: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        key = (height, width)
        cached = self._torch_grid_cache.get(key)
        if cached is not None:
            return cached

        assert torch is not None
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        cached = (grid_x, grid_y)
        self._torch_grid_cache[key] = cached
        return cached

    def _render_numpy(self, cloud: LoadedGaussianCloud, pose: CameraPose) -> np.ndarray:
        image = np.full((self.height, self.width, 3), 255.0, dtype=np.float32)

        centered = cloud.means - pose.eye[None, :]
        camera_points = centered @ pose.rotation_world_to_camera.T
        camera_covariances = np.einsum(
            "ij,njk,kl->nil",
            pose.rotation_world_to_camera,
            cloud.covariances,
            pose.rotation_world_to_camera.T,
            optimize=True,
        )

        x = camera_points[:, 0]
        y = camera_points[:, 1]
        z = camera_points[:, 2]
        visible = z > 1e-3
        if not np.any(visible):
            return image.astype(np.uint8)

        x = x[visible]
        y = y[visible]
        z = z[visible]
        colors = cloud.colors[visible]
        opacities = cloud.opacities[visible]
        camera_covariances = camera_covariances[visible]

        screen_x = self.fx * x / z + self.cx
        screen_y = self.cy - self.fy * y / z
        in_frame = screen_x >= -self.max_splat_radius
        in_frame &= screen_x < self.width + self.max_splat_radius
        in_frame &= screen_y >= -self.max_splat_radius
        in_frame &= screen_y < self.height + self.max_splat_radius
        if not np.any(in_frame):
            return image.astype(np.uint8)

        x = x[in_frame]
        y = y[in_frame]
        z = z[in_frame]
        screen_x = screen_x[in_frame]
        screen_y = screen_y[in_frame]
        colors = colors[in_frame]
        opacities = opacities[in_frame]
        camera_covariances = camera_covariances[in_frame]

        inv_z = 1.0 / z
        jacobian = np.zeros((len(z), 2, 3), dtype=np.float32)
        jacobian[:, 0, 0] = self.fx * inv_z
        jacobian[:, 0, 2] = -self.fx * x * inv_z * inv_z
        jacobian[:, 1, 1] = -self.fy * inv_z
        jacobian[:, 1, 2] = self.fy * y * inv_z * inv_z
        covariance_2d = np.einsum("nij,njk,nlk->nil", jacobian, camera_covariances, jacobian, optimize=True)
        covariance_2d[:, 0, 0] += 0.25
        covariance_2d[:, 1, 1] += 0.25

        eigenvalues = np.linalg.eigvalsh(covariance_2d)
        eigenvalues = np.clip(eigenvalues, 1e-6, None)
        radii = np.ceil(self.sigma_cutoff * np.sqrt(eigenvalues[:, 1])).astype(np.int32)
        radii = np.clip(radii, 1, self.max_splat_radius)
        inv_covariances = np.linalg.inv(covariance_2d)

        far_to_near = np.argsort(z)[::-1]

        cutoff_sq = self.sigma_cutoff * self.sigma_cutoff

        for index in far_to_near:
            radius = int(radii[index])
            screen_x_value = float(screen_x[index])
            screen_y_value = float(screen_y[index])

            min_x = max(0, int(math.floor(screen_x_value - radius)))
            max_x = min(self.width, int(math.ceil(screen_x_value + radius + 1)))
            min_y = max(0, int(math.floor(screen_y_value - radius)))
            max_y = min(self.height, int(math.ceil(screen_y_value + radius + 1)))
            if min_x >= max_x or min_y >= max_y:
                continue

            grid_x, grid_y = self._numpy_local_grid(max_y - min_y, max_x - min_x)
            delta_x = grid_x + min_x + 0.5 - screen_x_value
            delta_y = grid_y + min_y + 0.5 - screen_y_value
            inv_covariance = inv_covariances[index]
            mahalanobis = (
                inv_covariance[0, 0] * delta_x * delta_x
                + 2.0 * inv_covariance[0, 1] * delta_x * delta_y
                + inv_covariance[1, 1] * delta_y * delta_y
            )
            support = mahalanobis <= cutoff_sq
            if not np.any(support):
                continue

            alpha = opacities[index] * np.exp(-0.5 * mahalanobis)
            alpha *= support.astype(np.float32)
            if float(alpha.max()) <= 1e-4:
                continue

            roi = image[min_y:max_y, min_x:max_x]
            alpha_expanded = alpha[..., None]
            color = colors[index][None, None, :] * 255.0
            roi[:] = roi * (1.0 - alpha_expanded) + color * alpha_expanded

        return np.clip(image, 0.0, 255.0).astype(np.uint8)

    def _torch_cloud(self, cloud: LoadedGaussianCloud) -> TorchGaussianCloud:
        cached = self._torch_cloud_cache.get(cloud.path)
        if cached is not None:
            return cached

        assert torch is not None
        torch_cloud = TorchGaussianCloud(
            means=torch.from_numpy(np.ascontiguousarray(cloud.means)).to(device=self.device, dtype=torch.float32),
            colors=torch.from_numpy(np.ascontiguousarray(cloud.colors)).to(device=self.device, dtype=torch.float32),
            opacities=torch.from_numpy(np.ascontiguousarray(cloud.opacities)).to(device=self.device, dtype=torch.float32),
            covariances=torch.from_numpy(np.ascontiguousarray(cloud.covariances)).to(device=self.device, dtype=torch.float32),
        )
        self._torch_cloud_cache[cloud.path] = torch_cloud
        return torch_cloud

    def _render_torch(self, cloud: LoadedGaussianCloud, pose: CameraPose) -> np.ndarray:
        if torch is None:
            raise RuntimeError("CUDA rendering requires PyTorch to be installed.")

        cloud_tensors = self._torch_cloud(cloud)
        image = torch.full((self.height, self.width, 3), 255.0, dtype=torch.float32, device=self.device)

        eye = torch.as_tensor(pose.eye, dtype=torch.float32, device=self.device)
        rotation = torch.as_tensor(pose.rotation_world_to_camera, dtype=torch.float32, device=self.device)
        centered = cloud_tensors.means - eye.unsqueeze(0)
        camera_points = centered @ rotation.T
        camera_covariances = torch.einsum("ij,njk,kl->nil", rotation, cloud_tensors.covariances, rotation.T)

        x = camera_points[:, 0]
        y = camera_points[:, 1]
        z = camera_points[:, 2]
        visible = z > 1e-3
        if not bool(torch.any(visible)):
            return image.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()

        x = x[visible]
        y = y[visible]
        z = z[visible]
        colors = cloud_tensors.colors[visible]
        opacities = cloud_tensors.opacities[visible]
        camera_covariances = camera_covariances[visible]

        screen_x = self.fx * x / z + self.cx
        screen_y = self.cy - self.fy * y / z
        in_frame = screen_x >= -self.max_splat_radius
        in_frame &= screen_x < self.width + self.max_splat_radius
        in_frame &= screen_y >= -self.max_splat_radius
        in_frame &= screen_y < self.height + self.max_splat_radius
        if not bool(torch.any(in_frame)):
            return image.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()

        x = x[in_frame]
        y = y[in_frame]
        z = z[in_frame]
        screen_x = screen_x[in_frame]
        screen_y = screen_y[in_frame]
        colors = colors[in_frame]
        opacities = opacities[in_frame]
        camera_covariances = camera_covariances[in_frame]

        inv_z = torch.reciprocal(z)
        jacobian = torch.zeros((z.shape[0], 2, 3), dtype=torch.float32, device=self.device)
        jacobian[:, 0, 0] = self.fx * inv_z
        jacobian[:, 0, 2] = -self.fx * x * inv_z * inv_z
        jacobian[:, 1, 1] = -self.fy * inv_z
        jacobian[:, 1, 2] = self.fy * y * inv_z * inv_z
        covariance_2d = torch.einsum("nij,njk,nlk->nil", jacobian, camera_covariances, jacobian)
        covariance_2d[:, 0, 0] += 0.25
        covariance_2d[:, 1, 1] += 0.25

        eigenvalues = torch.linalg.eigvalsh(covariance_2d)
        eigenvalues = torch.clamp(eigenvalues, min=1e-6)
        radii = torch.ceil(self.sigma_cutoff * torch.sqrt(eigenvalues[:, 1])).to(torch.int32)
        radii = torch.clamp(radii, 1, self.max_splat_radius)
        inv_covariances = torch.linalg.inv(covariance_2d)

        far_to_near = torch.argsort(z, descending=True)
        cutoff_sq = self.sigma_cutoff * self.sigma_cutoff

        for index in far_to_near.tolist():
            radius = int(radii[index].item())
            screen_x_value = float(screen_x[index].item())
            screen_y_value = float(screen_y[index].item())

            min_x = max(0, int(math.floor(screen_x_value - radius)))
            max_x = min(self.width, int(math.ceil(screen_x_value + radius + 1)))
            min_y = max(0, int(math.floor(screen_y_value - radius)))
            max_y = min(self.height, int(math.ceil(screen_y_value + radius + 1)))
            if min_x >= max_x or min_y >= max_y:
                continue

            grid_x, grid_y = self._torch_local_grid(max_y - min_y, max_x - min_x)
            delta_x = grid_x + min_x + 0.5 - screen_x[index]
            delta_y = grid_y + min_y + 0.5 - screen_y[index]
            inv_covariance = inv_covariances[index]
            mahalanobis = (
                inv_covariance[0, 0] * delta_x * delta_x
                + 2.0 * inv_covariance[0, 1] * delta_x * delta_y
                + inv_covariance[1, 1] * delta_y * delta_y
            )
            support = mahalanobis <= cutoff_sq
            if not bool(torch.any(support)):
                continue

            alpha = opacities[index] * torch.exp(-0.5 * mahalanobis)
            alpha = alpha * support.to(dtype=torch.float32)
            if float(torch.max(alpha).item()) <= 1e-4:
                continue

            roi = image[min_y:max_y, min_x:max_x]
            alpha_expanded = alpha.unsqueeze(-1)
            color = colors[index].view(1, 1, 3) * 255.0
            roi.copy_(roi * (1.0 - alpha_expanded) + color * alpha_expanded)

        return image.clamp(0.0, 255.0).to(torch.uint8).cpu().numpy()


def build_video(
    clouds: list[LoadedGaussianCloud],
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    schedule: list[tuple[int, int, float]],
    start_azim: float,
    degrees_per_transition: float,
    elevation: float,
    roll: float,
    lower: np.ndarray,
    upper: np.ndarray,
    vertical_fov: float,
    sigma_cutoff: float,
    max_splat_radius: int,
    device: str,
    frame_workers: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*CODEC),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {output_path}")

    total_frames = len(schedule)
    total_rotation = degrees_per_transition * max(len(clouds) - 1, 1)
    scene_center = (lower + upper) / 2.0
    scene_radius = np.max(upper - lower) / 2.0
    distance = scene_radius / math.tan(math.radians(vertical_fov) / 2.0)
    distance *= 1.15
    estimated_direction = estimate_scene_stem_direction(clouds)
    stem_axis = resolve_stem_axis_for_scene(choose_stem_axis(clouds), lower, upper, estimated_direction)
    tasks = build_render_tasks(
        schedule=schedule,
        start_azim=start_azim,
        total_rotation=total_rotation,
        stem_axis=stem_axis,
        scene_center=scene_center,
        distance=distance,
        elevation=elevation,
        roll=roll,
    )

    try:
        if device == "cpu" and frame_workers > 1:
            max_workers = min(frame_workers, os.cpu_count() or frame_workers, len(tasks))
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=init_frame_worker,
                initargs=(
                    clouds,
                    width,
                    height,
                    vertical_fov,
                    sigma_cutoff,
                    max_splat_radius,
                    device,
                ),
            ) as executor:
                future_map = {
                    executor.submit(render_frame_task, task): task.frame_index
                    for task in tasks
                }

                completed_frames: dict[int, np.ndarray] = {}
                next_frame_to_write = 1
                for future in concurrent.futures.as_completed(future_map):
                    frame_index, frame = future.result()
                    completed_frames[frame_index] = frame
                    while next_frame_to_write in completed_frames:
                        ready_frame = completed_frames.pop(next_frame_to_write)
                        writer.write(cv2.cvtColor(ready_frame, cv2.COLOR_RGB2BGR))
                        print_progress(next_frame_to_write, total_frames, "Rendering video")
                        next_frame_to_write += 1
        else:
            renderer = GaussianSplatRenderer(
                width=width,
                height=height,
                vertical_fov_deg=vertical_fov,
                sigma_cutoff=sigma_cutoff,
                max_splat_radius=max_splat_radius,
                device=device,
            )
            for task in tasks:
                left_cloud = clouds[task.left_index]
                right_cloud = clouds[task.right_index]
                left_frame = renderer.render(left_cloud, task.pose)
                if task.left_index == task.right_index or task.alpha <= 0.0:
                    frame = left_frame
                else:
                    right_frame = renderer.render(right_cloud, task.pose)
                    frame = blend_frames(left_frame, right_frame, task.alpha)

                frame = add_overlay(frame, frame_label(left_cloud, right_cloud, task.alpha))
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print_progress(task.frame_index, total_frames, "Rendering video")
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    render_device = resolve_render_device(args.device)
    print(f"Rendering backend: {render_device}")
    if args.frame_workers < 1:
        raise ValueError("--frame-workers must be at least 1")
    if render_device != "cpu" and args.frame_workers > 1:
        print("Frame workers are only used for CPU rendering; using a single renderer process for this backend.")
    transition_frames = max(1, int(round(args.seconds_per_transition * args.fps)))
    hold_frames = max(0, int(round(args.hold_seconds * args.fps)))

    if not args.skip_axis_sync:
        if args.axis_file.is_file():
            sync_axis_file_to_plys(args.axis_file, args.data_dir)
        else:
            print(f"Axis file not found, keeping existing PLY header axis metadata: {args.axis_file}")

    clouds = discover_point_clouds(
        args.data_dir,
        min_opacity=args.min_opacity,
        splat_scale=args.splat_scale,
    )
    lower, upper = global_scene_limits(clouds, padding_fraction=args.padding)
    schedule = build_frame_schedule(
        cloud_count=len(clouds),
        transition_frames=transition_frames,
        hold_frames=hold_frames,
    )
    build_video(
        clouds=clouds,
        output_path=args.output,
        fps=args.fps,
        width=args.width,
        height=args.height,
        schedule=schedule,
        start_azim=args.start_azim,
        degrees_per_transition=args.degrees_per_transition,
        elevation=args.elevation,
        roll=args.roll,
        lower=lower,
        upper=upper,
        vertical_fov=args.vertical_fov,
        sigma_cutoff=args.sigma_cutoff,
        max_splat_radius=args.max_splat_radius,
        device=render_device,
        frame_workers=args.frame_workers,
    )
    print(f"Saved video to: {args.output}")


if __name__ == "__main__":
    main()
