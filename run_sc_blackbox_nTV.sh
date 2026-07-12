#!/bin/bash
# Phase 2: Blackbox nTV ablation — Speech Commands (runs after whitebox finishes)
set -e

declare -A GPU_MAP=( [0]=0 [0.5]=1 [1]=2 [1.5]=3 [2]=5 )
LOG_DIR=/tmp/sc_blackbox_nTV_logs
mkdir -p "$LOG_DIR"

SCRIPT=/media/sda1/zxlong/tmc_re/sc_blackbox_attack_nTV.py

echo "[$(date)] Launching blackbox nTV training..."

for NTV in 0 0.5 1 1.5 2; do
    GPU=${GPU_MAP[$NTV]}
    LOG="$LOG_DIR/blackbox_nTV${NTV}.log"

    nohup bash -c "
        export OMP_NUM_THREADS=2
        for SP in 1 2 3 4; do
            echo \"[\$(date)] blackbox nTV=$NTV GPU=$GPU sp\$SP starting...\"
            python '$SCRIPT' --nTV $NTV --cuda_id $GPU --split_point \$SP --epochs 100 --batch-size 512 --log-interval 50 --num_workers 0
            echo \"[\$(date)] blackbox nTV=$NTV GPU=$GPU sp\$SP done.\"
        done
        echo \"[\$(date)] All split points complete for nTV=$NTV.\"
    " > "$LOG" 2>&1 &
    sleep 3
done

echo "[$(date)] All blackbox jobs launched."
echo "Monitor: tail -f $LOG_DIR/blackbox_nTV*.log"
