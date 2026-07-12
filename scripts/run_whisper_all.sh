#!/bin/bash
set -e

EXPS=(
  "4 whitebox 0"
  "4 blackbox 1"
  "8 whitebox 2"
  "8 blackbox 3"
)

for exp in "${EXPS[@]}"; do
  IFS=' ' read -r sp mode dev <<< "$exp"
  LOG="whisper_sp${sp}_${mode}.log"
  CUDA_VISIBLE_DEVICES=$dev python scripts/train_whisper_inversion.py \
    --split_point $sp \
    --mode $mode \
    --epochs 100 \
    --batch_size 8 \
    --device_id 0 \
    > "$LOG" 2>&1 &
  echo "Launched sp=$sp mode=$mode on GPU $dev (log=$LOG)"
done

wait
echo "All experiments done."
