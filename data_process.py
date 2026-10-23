"""Prepare legacy training crops without executing work at import time."""
import argparse
from pathlib import Path
import sys

import numpy as np

from ptoart.io import SUPPORTED, read_image, write_image


def main(argv=None):
    parser = argparse.ArgumentParser(description="ptoart training image preprocessing")
    parser.add_argument("--data_path", required=True, type=Path)
    parser.add_argument("--save_path", required=True, type=Path)
    parser.add_argument("--segmentation", type=Path,
                        default=Path(__file__).resolve().parent / "utils/seg_model_384.pb")
    args = parser.parse_args(argv)
    source = args.data_path.resolve()
    destination = args.save_path.resolve()
    if not source.is_dir() or source == destination or source.is_relative_to(destination):
        parser.error("Input must be a directory distinct from and outside the output directory")
    files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED)
    from utils import Preprocess
    failed = 0
    with Preprocess(seg_model_path=str(args.segmentation)) as pre:
        for index, path in enumerate(files):
            try:
                rgba = pre.process(read_image(path))
                if rgba is None:
                    raise ValueError("No face detected")
                mask = rgba[:, :, 3:4].astype(np.float32) / 255
                face = (rgba[:, :, :3] * mask + (1 - mask) * 255).astype(np.uint8)
                write_image(destination / f"{index:04d}.png", face)
            except Exception as exc:
                failed += 1
                print(f"{path}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
