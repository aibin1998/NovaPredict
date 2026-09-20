#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/data/wangab/miniforge3/envs/casanovo_env2/bin/python
OUT_DIR="$HERE/log_full_direct_pimeancov_lambda0p01_bs2048_seed1337"

cd "$HERE"
env CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=4 "$PYTHON" -u train_decoder.py \
  --device cuda:0 \
  --epochs 30 \
  --patience 10 \
  --batch-size 2048 \
  --shuffle-block-size 2048 \
  --num-workers 4 \
  --prefetch-factor 1 \
  --decoder-learning-rate 5e-4 \
  --stage1-backbone-learning-rate 5e-5 \
  --stage1-head-learning-rate 1e-4 \
  --minimum-learning-rate 1e-5 \
  --stage2-distribution-weight 0.01 \
  --w-sa 1.0 \
  --w-mvn 0.1 \
  --w-bce 0.5 \
  --hidden-size 192 \
  --residual-blocks 3 \
  --out-dir "$OUT_DIR"
