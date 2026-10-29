"""CUDA entry point for the original charge-aggregated LogMVN pretraining."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.logmvn_support.train import main  # noqa: E402


if __name__ == "__main__":
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
    if device.startswith("cuda") and (torch.version.cuda is None or torch.version.hip is not None):
        raise RuntimeError("The NVIDIA entry point requires a CUDA PyTorch build")
    main()
