#!/usr/bin/env python3
"""Write a runnable 4DGS YAML config with local data/output paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--loaded-pth", default="")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.template)
    cfg.ModelParams.source_path = args.source_path
    cfg.ModelParams.model_path = args.model_path
    cfg.ModelParams.loaded_pth = args.loaded_pth

    args.output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=args.output)
    print(args.output)


if __name__ == "__main__":
    main()
