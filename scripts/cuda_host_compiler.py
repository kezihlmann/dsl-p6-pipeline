from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path


SUPPORTED_GCC_MAJOR = 12


def _read_compiler_major(compiler: str) -> int | None:
    commands = ([compiler, "-dumpfullversion"], [compiler, "--version"])
    for command in commands:
        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        match = re.search(r"(\d+)(?:\.\d+)?(?:\.\d+)?", completed.stdout)
        if match:
            return int(match.group(1))
    return None


def _find_supported_pair() -> tuple[str, str] | None:
    candidates = [
        ("gcc-12", "g++-12"),
        ("gcc-11", "g++-11"),
        ("gcc-10", "g++-10"),
        ("gcc", "g++"),
    ]
    for cc_name, cxx_name in candidates:
        cc_path = shutil.which(cc_name)
        cxx_path = shutil.which(cxx_name)
        if cc_path is None or cxx_path is None:
            continue
        cc_major = _read_compiler_major(cc_path)
        cxx_major = _read_compiler_major(cxx_path)
        if cc_major is None or cxx_major is None:
            continue
        if cc_major != cxx_major:
            continue
        if cc_major <= SUPPORTED_GCC_MAJOR:
            return cc_path, cxx_path
    return None


def configure_cuda_host_compiler(env: dict[str, str]) -> tuple[str, str]:
    cc = env.get("CC")
    cxx = env.get("CXX")

    if cc and cxx:
        cc_major = _read_compiler_major(cc)
        cxx_major = _read_compiler_major(cxx)
        if cc_major is None or cxx_major is None:
            raise RuntimeError(
                f"Could not detect versions for CC={cc!r} and CXX={cxx!r}. "
                "Set them to a GCC/G++ pair supported by nvcc."
            )
        if cc_major != cxx_major or cc_major > SUPPORTED_GCC_MAJOR:
            raise RuntimeError(
                f"CC={cc!r} and CXX={cxx!r} resolve to GCC {cc_major}/{cxx_major}, "
                f"but CUDA on Euler needs GCC {SUPPORTED_GCC_MAJOR}.x or older for native builds."
            )
        env["CUDAHOSTCXX"] = cxx
        return cc, cxx

    pair = _find_supported_pair()
    if pair is None:
        raise RuntimeError(
            "Could not find a supported GCC/G++ host compiler for nvcc. "
            f"Load a GCC {SUPPORTED_GCC_MAJOR}.x (or older) module on Euler, then rerun this helper. "
            "If you already know the paths, export CC, CXX, and optionally CUDAHOSTCXX first."
        )

    cc, cxx = pair
    env["CC"] = cc
    env["CXX"] = cxx
    env["CUDAHOSTCXX"] = cxx
    return cc, cxx


def describe_compiler_pair(cc: str, cxx: str) -> str:
    cc_major = _read_compiler_major(cc)
    cxx_major = _read_compiler_major(cxx)
    version_text = "unknown"
    if cc_major is not None and cxx_major is not None:
        version_text = f"GCC {cc_major}/{cxx_major}"
    return f"{version_text} via CC={Path(cc).name}, CXX={Path(cxx).name}"
