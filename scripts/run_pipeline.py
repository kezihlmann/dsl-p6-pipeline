from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_BY_STEP = {
    "step_1": "create_sam3_masks.py",
    "step_3": "create_video.py",
}

STEP_2_SCRIPT_BY_METHOD = {
    "3dgs": "create_3dgs_reconstructions.py",
    "nerfacto": "create_nerfacto_reconstructions.py",
}

DEFAULT_ENV_BY_RECONSTRUCTION_METHOD = {
    "3dgs": "dsl-p6-pipeline",
    "nerfacto": "dsl-p6-nerfacto",
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


def resolve_conda_executable() -> str | None:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        return conda_exe
    return shutil.which("conda")


def build_command_for_step(
    step_name: str,
    script_path: Path,
    settings_path: Path,
    settings: dict[str, object],
) -> list[str]:
    if step_name != "step_2":
        return [sys.executable, str(script_path), "--settings", str(settings_path)]

    reconstruction_method = str(settings.get("reconstruction_method", "3dgs")).strip().lower()
    env_name = DEFAULT_ENV_BY_RECONSTRUCTION_METHOD[reconstruction_method]

    if reconstruction_method == "3dgs":
        return [sys.executable, str(script_path), "--settings", str(settings_path)]

    conda_executable = resolve_conda_executable()
    if conda_executable is None:
        raise RuntimeError(
            "Could not find the conda executable. "
            "Activate the base conda setup before running the pipeline so step 2 can switch into the Nerfacto environment automatically."
        )

    return [
        conda_executable,
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        str(script_path),
        "--settings",
        str(settings_path),
    ]


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

    for step_name in ("step_1", "step_2", "step_3"):
        if not bool(settings.get(step_name, False)):
            print(f"Skipping {step_name}: disabled in settings.")
            continue

        if step_name == "step_2":
            reconstruction_method = str(settings.get("reconstruction_method", "3dgs")).strip().lower()
            script_name = STEP_2_SCRIPT_BY_METHOD.get(reconstruction_method)
            if script_name is None:
                raise ValueError(
                    f"Unsupported reconstruction_method={reconstruction_method!r}. "
                    f"Expected one of: {', '.join(sorted(STEP_2_SCRIPT_BY_METHOD))}."
                )
        else:
            script_name = SCRIPT_BY_STEP[step_name]

        script_path = scripts_dir / script_name
        command = build_command_for_step(
            step_name=step_name,
            script_path=script_path,
            settings_path=settings_path,
            settings=settings,
        )
        print(f"Running {step_name}: {' '.join(command)}")
        subprocess.run(command, cwd=repo_root, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
