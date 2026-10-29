"""ROCm entry point for charge-aggregated LogMVN pretraining."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.logmvn_support.train import main  # noqa: E402


if __name__ == "__main__":
    # On the tested ROCm stack, MIOpen's two-layer LSTM backward can silently
    # zero the encoder gradient. The native PyTorch implementation is required.
    torch.backends.cudnn.enabled = False
    if "--check-encoder-gradient" not in sys.argv:
        sys.argv.append("--check-encoder-gradient")
    device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cuda:0"
    if device.startswith("cuda") and torch.version.hip is None:
        raise RuntimeError("The AMD entry point requires a ROCm PyTorch build")
    main()
