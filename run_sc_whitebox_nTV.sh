#!/bin/bash
# Phase 1: Whitebox nTV ablation — Speech Commands (uses cached dataset)
set -e

declare -A GPU_MAP=( [0]=0 [0.5]=1 [1]=2 [1.5]=3 [2]=5 )
LOG_DIR=/tmp/sc_whitebox_nTV_logs
mkdir -p "$LOG_DIR"

SCRIPT=/media/sda1/zxlong/tmc_re/sc_whitebox_attack_nTV.py

for NTV in 0 0.5 1 1.5 2; do
    GPU=${GPU_MAP[$NTV]}
    LOG="$LOG_DIR/whitebox_nTV${NTV}.log"
    echo "[$(date)] Launching nTV=$NTV on GPU=$GPU -> $LOG"

    nohup bash -c "
        export OMP_NUM_THREADS=2
        for SP in 1 2 3 4; do
            echo \"[\$(date)] nTV=$NTV GPU=$GPU sp\$SP starting...\"
            python '$SCRIPT' --nTV $NTV --cuda_id $GPU --split_point \$SP --epochs 100 --batch-size 128 --log-interval 50 --num_workers 0
            echo \"[\$(date)] nTV=$NTV GPU=$GPU sp\$SP done.\"
        done
        echo \"[\$(date)] All split points complete for nTV=$NTV.\"
    " > "$LOG" 2>&1 &

    sleep 2
done

echo "[$(date)] All whitebox jobs launched."
echo "Monitor: tail -f $LOG_DIR/whitebox_nTV*.log"
