#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data/logmvn}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-${PYTHON_BIN:-python}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"

exec "$PYTHON" "$HERE/pretrain_cnn_bilstm.py" \
  --train-hdf5 "${TRAIN_HDF5:-$DATA_DIR/train.hdf5}" \
  --val-hdf5 "${VAL_HDF5:-$DATA_DIR/val.hdf5}" \
  --epochs 50 \
  --detail-interval 5 \
  --device "${DEVICE:-cuda:0}" \
  --out-dir "${OUT_DIR:-$ROOT/outputs/nvidia/pretrain_cnn_bilstm}" \
  "$@"
