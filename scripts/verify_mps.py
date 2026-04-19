#!/usr/bin/env python3
"""Verify Python, torch, and Apple Silicon MPS availability."""

from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    try:
        import torch
    except ImportError:
        print("Torch: not installed")
        return 1

    print(f"Torch: {torch.__version__}")
    built = torch.backends.mps.is_built()
    available = torch.backends.mps.is_available()
    print(f"MPS built: {built}")
    print(f"MPS available: {available}")
    if available:
        x = torch.ones(3, device="mps")
        print(f"Tensor on MPS: {x}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

