#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-0}"
TRAIN_HDF5="${TRAIN_HDF5:-$ROOT/data/train_multi_charge.hdf5}"
VAL_HDF5="${VAL_HDF5:-$ROOT/data/val_multi_charge.hdf5}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/amd}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-30}"

for path in "$TRAIN_HDF5" "$VAL_HDF5"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing input HDF5: $path" >&2
    exit 1
  fi
done

init_args=()
if [[ "${USE_LOGMVN_PRETRAIN:-1}" == "1" ]]; then
  INIT_CHECKPOINT="${INIT_CHECKPOINT:-$ROOT/weights/logmvn_best.pt}"
  if [[ ! -f "$INIT_CHECKPOINT" ]]; then
    echo "Missing LogMVN initialization checkpoint: $INIT_CHECKPOINT" >&2
    exit 1
  fi
  init_args=(--init-checkpoint "$INIT_CHECKPOINT")
fi

# PyTorch ROCm exposes its GPU as cuda:0. The tested ROCm 7.1 stack needs
# serialized HIP kernels; both trainers use FP32 and the native LSTM backend.
env ROCR_VISIBLE_DEVICES="$GPU" AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-3}" \
  "$PYTHON" "$HERE/train.py" \
  --train-hdf5 "$TRAIN_HDF5" --val-hdf5 "$VAL_HDF5" \
  --out-dir "$OUTPUT_DIR/stage1" --device cuda:0 --no-amp \
  --batch-size "$BATCH_SIZE" --shuffle-block-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" --stage1-epochs "$STAGE1_EPOCHS" \
  --stage2-epochs 0 "${init_args[@]}"

env ROCR_VISIBLE_DEVICES="$GPU" AMD_SERIALIZE_KERNEL="${AMD_SERIALIZE_KERNEL:-3}" \
  "$PYTHON" "$HERE/train_decoder.py" \
  --train-hdf5 "$TRAIN_HDF5" --val-hdf5 "$VAL_HDF5" \
  --stage1-checkpoint "$OUTPUT_DIR/stage1/stage1_best.pt" \
  --out-dir "$OUTPUT_DIR/stage2" --device cuda:0 --no-amp \
  --batch-size "$BATCH_SIZE" --shuffle-block-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" --epochs "$STAGE2_EPOCHS" \
  --stage2-distribution-weight 0.01
