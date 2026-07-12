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
  LOG="eval_sp${sp}_${mode}.log"
  CUDA_VISIBLE_DEVICES=$dev python scripts/eval_whisper_inversion.py \
    --split_point $sp \
    --mode $mode \
    --device_id 0 \
    > "$LOG" 2>&1 &
  echo "Launched eval sp=$sp mode=$mode on GPU $dev (log=$LOG)"
done

echo "Waiting for all evals to finish..."
wait
echo "All evals done."
