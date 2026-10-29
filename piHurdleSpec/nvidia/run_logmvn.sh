#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
TRAIN_HDF5="${LOGMVN_TRAIN_HDF5:-$ROOT/data/logmvn/train.hdf5}"
VAL_HDF5="${LOGMVN_VAL_HDF5:-$ROOT/data/logmvn/val.hdf5}"
OUT_DIR="${LOGMVN_OUT_DIR:-$ROOT/outputs/nvidia/logmvn}"
INIT_CHECKPOINT="${DETERMINISTIC_CHECKPOINT:-$ROOT/weights/cnn_bilstm_best.pt}"

for path in "$TRAIN_HDF5" "$VAL_HDF5" "$INIT_CHECKPOINT"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing pretraining input: $path" >&2
    exit 1
  fi
done

env CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$HERE/pretrain_logmvn.py" \
  --train-hdf5 "$TRAIN_HDF5" --val-hdf5 "$VAL_HDF5" \
  --init-checkpoint "$INIT_CHECKPOINT" --covariance-rank 8 \
  --head-warmup-epochs 2 --epochs 50 --detail-interval 5 \
  --device cuda:0 --out-dir "$OUT_DIR"
