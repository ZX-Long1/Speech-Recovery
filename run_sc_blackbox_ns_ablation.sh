#!/bin/bash
# num_samples ablation: {10,20,30,40} × sp={1,3}, nTV=0.5
set -e

LOG_DIR=/tmp/sc_blackbox_ns_ablation_logs
rm -rf "$LOG_DIR"
mkdir -p "$LOG_DIR"

SCRIPT=/media/sda1/zxlong/tmc_re/speech_command_blackbox_attack1.py
NS_VALUES=(10 20 30 40)
GPU_MAP=(0 1 2 3)
NTV=0.5

echo "[$(date)] Launching SC blackbox num_samples ablation..."

for i in "${!NS_VALUES[@]}"; do
    NS="${NS_VALUES[$i]}"
    GPU="${GPU_MAP[$i]}"
    LOG="$LOG_DIR/ns${NS}.log"

    nohup bash -c "
        export OMP_NUM_THREADS=2
        SP=1
        echo \"[\$(date)] ns=${NS} sp=\${SP} GPU=${GPU} starting...\"
        python -u '$SCRIPT' --cuda_id ${GPU} --split_point \${SP} --num_samples ${NS} --nTV ${NTV} --output_tag _ns${NS} --epochs 100 --batch-size 128 --log-interval 50 --num_workers 0
        echo \"[\$(date)] ns=${NS} sp=\${SP} done.\"

        SP=3
        echo \"[\$(date)] ns=${NS} sp=\${SP} GPU=${GPU} starting...\"
        python -u '$SCRIPT' --cuda_id ${GPU} --split_point \${SP} --num_samples ${NS} --nTV ${NTV} --output_tag _ns${NS} --epochs 100 --batch-size 128 --log-interval 50 --num_workers 0
        echo \"[\$(date)] ns=${NS} sp=\${SP} done.\"

        echo \"[\$(date)] All complete for ns=${NS}.\"
    " > "$LOG" 2>&1 &
    sleep 10
done

echo "[$(date)] All jobs launched."
echo "Monitor: tail -f $LOG_DIR/ns*.log"
