"""Train the deterministic CNN-BiLSTM from random initialization on AMD ROCm."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent / "deterministic_pretrain"
sys.path.insert(0, str(SOURCE_DIR))
runpy.run_path(str(SOURCE_DIR / "train.py"), run_name="__main__")
