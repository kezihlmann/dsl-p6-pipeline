from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path


SCRIPT_BY_STEP = {
    "step_1": "create_sam3_masks.py",
    "step_2": "create_colmap.py",
    "step_3": "create_3dgs_reconstructions.py",
    "step_4": "create_video.py",
}


def parse_settings(settings_path: Path) -> dict[str, object]:
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
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description="Run enabled pipeline stages from settings_pipeline.txt.")
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "settings_pipeline.txt",
        help="Path to the pipeline settings file.",
    )
    args = parser.parse_args()

    settings_path = args.settings.resolve()
    repo_root = settings_path.parent
    scripts_dir = Path(__file__).resolve().parent
    settings = parse_settings(settings_path)

    for step_name, script_name in SCRIPT_BY_STEP.items():
        if not bool(settings.get(step_name, False)):
            print(f"Skipping {step_name}: disabled in settings.")
            continue

        script_path = scripts_dir / script_name
        command = [sys.executable, str(script_path), "--settings", str(settings_path)]
        print(f"Running {step_name}: {' '.join(command)}")
        subprocess.run(command, cwd=repo_root, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())