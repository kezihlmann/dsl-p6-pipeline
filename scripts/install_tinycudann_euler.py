from __future__ import annotations

import os
import shutil
import subprocess
import sys

from cuda_host_compiler import configure_cuda_host_compiler, describe_compiler_pair


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def run(command: list[str], env: dict[str, str]) -> None:
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True, env=env)


def main() -> int:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        return fail(f"Failed to import torch: {exc}")

    if shutil.which("nvcc") is None:
        return fail("nvcc was not found on PATH. Load the CUDA module on a GPU node before running this helper.")

    if not torch.cuda.is_available():
        return fail(
            "torch.cuda.is_available() is False. Run this helper inside the dsl-p6-nerfacto environment on a GPU node."
        )

    capability = torch.cuda.get_device_capability()
    architecture = f"{capability[0]}{capability[1]}"

    env = os.environ.copy()
    env.setdefault("TCNN_CUDA_ARCHITECTURES", architecture)

    cuda_home = env.get("CUDA_HOME")
    if not cuda_home:
        nvcc_path = shutil.which("nvcc")
        if nvcc_path is None:
            return fail("nvcc disappeared from PATH.")
        env["CUDA_HOME"] = os.path.dirname(os.path.dirname(os.path.realpath(nvcc_path)))
    env.setdefault("CUDACXX", os.path.join(env["CUDA_HOME"], "bin", "nvcc"))
    cc, cxx = configure_cuda_host_compiler(env)

    print(f"Detected CUDA architecture: {env['TCNN_CUDA_ARCHITECTURES']}")
    print(f"Using CUDA_HOME={env['CUDA_HOME']}")
    print(f"Using host compiler {describe_compiler_pair(cc, cxx)}")
    print(f"Using torch {torch.__version__} with CUDA {torch.version.cuda}")

    run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--force-reinstall",
         "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"], env)

    import tinycudann  # noqa: F401

    print("tinycudann ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
