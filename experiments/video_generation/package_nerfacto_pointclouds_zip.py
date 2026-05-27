from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


DEFAULT_REPO_ROOT = Path("/cluster/project/cropsci/kzihlmann/dsl-p6-pipeline")
DEFAULT_OUTPUT_ZIP = DEFAULT_REPO_ROOT / "experiments" / "video_generation" / "pointclouds_maize_4.zip"
DEFAULT_START_TIMESTEP = 0
DEFAULT_END_TIMESTEP = 3300
DEFAULT_TIMESTEP_STEP = 60
DEFAULT_RESOLUTION_DECREASE_FACTOR = 4
DEFAULT_NUM_ITERATIONS = 30000
DEFAULT_POINTCLOUD_ITERATION = 10000
DEFAULT_MIN_FILE_AGE_SECONDS = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run on the Euler login node. Collect ready Nerfacto point clouds from data/maize_4 "
            "and package them into one zip file with names like point_cloud_0000.ply."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help=f"Cluster repo root. Default: {DEFAULT_REPO_ROOT}",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=DEFAULT_OUTPUT_ZIP,
        help=f"Output zip path. Default: {DEFAULT_OUTPUT_ZIP}",
    )
    parser.add_argument("--start", type=int, default=DEFAULT_START_TIMESTEP)
    parser.add_argument("--end", type=int, default=DEFAULT_END_TIMESTEP)
    parser.add_argument("--step", type=int, default=DEFAULT_TIMESTEP_STEP)
    parser.add_argument(
        "--resolution-decrease-factor",
        type=int,
        default=DEFAULT_RESOLUTION_DECREASE_FACTOR,
    )
    parser.add_argument("--num-iterations", type=int, default=DEFAULT_NUM_ITERATIONS)
    parser.add_argument("--pointcloud-iteration", type=int, default=DEFAULT_POINTCLOUD_ITERATION)
    parser.add_argument(
        "--min-file-age-seconds",
        type=int,
        default=DEFAULT_MIN_FILE_AGE_SECONDS,
        help=(
            "Only include point clouds older than this threshold to avoid files still being written. "
            f"Default: {DEFAULT_MIN_FILE_AGE_SECONDS}"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output zip file.",
    )
    return parser.parse_args()


def iter_timesteps(start: int, end: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("--step must be at least 1.")
    if end < start:
        raise ValueError("--end must be greater than or equal to --start.")
    return list(range(start, end + 1, step))


def build_pointcloud_path(
    repo_root: Path,
    timestep: int,
    resolution_decrease_factor: int,
    num_iterations: int,
    pointcloud_iteration: int,
) -> Path:
    timestep_name = f"timestep_{timestep:04d}"
    experiment_name = f"nerfacto_rgba_{timestep_name}_down{resolution_decrease_factor}_{num_iterations}"
    return (
        repo_root
        / "data"
        / "maize_4"
        / timestep_name
        / "nerfacto-reconstructions"
        / experiment_name
        / f"pointcloud_{pointcloud_iteration}"
        / "point_cloud.ply"
    )


def is_old_enough(path: Path, min_file_age_seconds: int) -> bool:
    age_seconds = path.stat().st_mtime
    import time

    return (time.time() - age_seconds) >= min_file_age_seconds


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_zip = args.output_zip.resolve()

    if not repo_root.is_dir():
        raise FileNotFoundError(f"Repo root does not exist: {repo_root}")
    if output_zip.exists() and not args.overwrite:
        raise FileExistsError(f"Output zip already exists: {output_zip}. Use --overwrite to replace it.")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    timesteps = iter_timesteps(args.start, args.end, args.step)

    added = 0
    missing = 0
    too_new = 0

    with ZipFile(output_zip, mode="w", compression=ZIP_DEFLATED) as archive:
        for timestep in timesteps:
            source_path = build_pointcloud_path(
                repo_root=repo_root,
                timestep=timestep,
                resolution_decrease_factor=args.resolution_decrease_factor,
                num_iterations=args.num_iterations,
                pointcloud_iteration=args.pointcloud_iteration,
            )
            archive_name = f"point_cloud_{timestep:04d}.ply"

            print(f"Checking timestep_{timestep:04d} ...")
            if not source_path.is_file():
                print(f"  Missing: {source_path}")
                missing += 1
                continue
            if not is_old_enough(source_path, args.min_file_age_seconds):
                print(f"  Skipping for now, still too new: {source_path}")
                too_new += 1
                continue

            archive.write(source_path, arcname=archive_name)
            print(f"  Added -> {archive_name}")
            added += 1

    print()
    print(f"Wrote zip file: {output_zip}")
    print(f"Added: {added}")
    print(f"Missing: {missing}")
    print(f"Too new: {too_new}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
