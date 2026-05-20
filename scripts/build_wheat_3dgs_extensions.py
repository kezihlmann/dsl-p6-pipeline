from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


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


def build_environment(repo_dir: Path) -> dict[str, str]:
    env = os.environ.copy()

    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        raise RuntimeError(
            "Could not find 'nvcc' on PATH. Load the CUDA module first before building Wheat-3DGS extensions."
        )

    cuda_home = Path(nvcc_path).resolve().parent.parent
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDACXX"] = str(cuda_home / "bin" / "nvcc")
    env["FORCE_CUDA"] = "1"
    env["PATH"] = os.pathsep.join([str(cuda_home / "bin"), env.get("PATH", "")]).rstrip(os.pathsep)

    import torch

    if torch.version.cuda is None:
        raise RuntimeError(
            "The active PyTorch build does not have CUDA enabled. "
            "Activate a CUDA-enabled dsl-p6-pipeline environment before building Wheat-3DGS extensions."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. Build Wheat-3DGS extensions on a GPU node, not on the Euler login node."
        )

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    ld_library_path_entries = [str(cuda_home / "lib64"), str(torch_lib)]
    existing_ld_library_path = env.get("LD_LIBRARY_PATH")
    if existing_ld_library_path:
        ld_library_path_entries.append(existing_ld_library_path)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_library_path_entries)

    python_path_entries = [str(repo_dir)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        python_path_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    return env


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build_extension(path: Path, env: dict[str, str], clean: bool) -> None:
    build_dir = path / "build"
    if clean and build_dir.exists():
        shutil.rmtree(build_dir)
    run_command([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=path, env=env)


def verify_simple_knn(repo_dir: Path, env: dict[str, str]) -> None:
    inline_code = "from simple_knn._C import distCUDA2; print('simple-knn ok')"
    run_command([sys.executable, "-c", inline_code], cwd=repo_dir, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch and build the Wheat-3DGS CUDA extensions from the parent repository."
    )
    parser.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "external" / "Wheat-3DGS",
        help="Path to the Wheat-3DGS submodule.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Keep existing build directories instead of removing them before compilation.",
    )
    args = parser.parse_args()

    repo_dir = args.repo_dir.resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"Wheat-3DGS repository not found: {repo_dir}")

    patched_files = patch_extension_setup_files(repo_dir)
    if patched_files:
        print(f"Patched extension setup files: {patched_files}")
    else:
        print("Extension setup files already patched.")

    env = build_environment(repo_dir)
    print(f"Using CUDA_HOME={env['CUDA_HOME']}")

    build_extension(repo_dir / "submodules" / "simple-knn", env=env, clean=not args.no_clean)
    build_extension(repo_dir / "submodules" / "diff-gaussian-rasterization", env=env, clean=not args.no_clean)
    build_extension(repo_dir / "submodules" / "flashsplat-rasterization", env=env, clean=not args.no_clean)
    verify_simple_knn(repo_dir, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
