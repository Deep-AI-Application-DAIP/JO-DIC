"""Top-level main entry for the JO-DIC reproduction package.

Usage examples:

    python main.py demo
    python main.py train --epochs 2 --batch-size 2 --height 128 --width 160 --max-disp 48
    python main.py build-data --out synthetic_speckle_demo --n 16
    python main.py info
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_demo() -> None:
    from jodic.demo_forward import main as demo_main
    demo_main()


def run_train(extra_args: list[str]) -> None:
    from jodic.train import main as train_main
    old_argv = sys.argv[:]
    try:
        sys.argv = ["jodic.train"] + extra_args
        train_main()
    finally:
        sys.argv = old_argv


def run_build_data(extra_args: list[str]) -> None:
    from tools.build_synthetic_dataset import main as build_main
    old_argv = sys.argv[:]
    try:
        sys.argv = ["tools.build_synthetic_dataset"] + extra_args
        build_main()
    finally:
        sys.argv = old_argv


def show_info() -> None:
    print("JO-DIC Reproduction Package")
    print("- Stereo branch: CGI-Stereo-like disparity estimation")
    print("- Flow branch: RAFT-like iterative optical-flow refinement")
    print("- Feature enhancement: LDConv / Linear Deformable Convolution")
    print("- Attention: Manhattan Self-Attention / Decomposed MaSA")
    print("- Loss: disparity + flow + spatial-temporal geometric consistency")
    print("- Output: 3D displacement and strain-field reconstruction")
    print()
    print("Common commands:")
    print("  python main.py demo")
    print("  python main.py train --epochs 2 --batch-size 2")
    print("  python main.py build-data --out synthetic_speckle_demo --n 16")


def parse_main_args() -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(
        description="Main entry for JO-DIC reproduction",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["demo", "train", "build-data", "info"],
        help=(
            "demo       Run one forward pass test\n"
            "train      Train on synthetic speckle data\n"
            "build-data Generate a small synthetic speckle dataset\n"
            "info       Print package information"
        ),
    )
    args, extra = parser.parse_known_args()
    return args.command, extra


def main() -> None:
    command, extra = parse_main_args()
    if command == "demo":
        run_demo()
    elif command == "train":
        run_train(extra)
    elif command == "build-data":
        run_build_data(extra)
    elif command == "info":
        show_info()
    else:
        raise ValueError(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
