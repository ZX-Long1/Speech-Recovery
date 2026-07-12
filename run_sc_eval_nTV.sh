#!/bin/bash
# Phase 3: Evaluate nTV ablation — Speech Commands (clean only)
set -e

declare -A GPU_MAP=( [0]=0 [0.5]=1 [1]=2 [1.5]=3 [2]=5 )
LOG_DIR=/tmp/sc_eval_nTV_logs
mkdir -p "$LOG_DIR"

SCRIPT=/media/sda1/zxlong/tmc_re/sc_evaluation_nTV.py

echo "[$(date)] Launching nTV evaluations..."

for NTV in 0 0.5 1 1.5 2; do
    GPU=${GPU_MAP[$NTV]}
    LOG="$LOG_DIR/eval_nTV${NTV}.log"

    nohup python "$SCRIPT" --nTV "$NTV" --gpu "$GPU" > "$LOG" 2>&1 &
    sleep 2
done

echo "[$(date)] All evaluation jobs launched."
echo "Monitor: tail -f $LOG_DIR/eval_nTV*.log"
