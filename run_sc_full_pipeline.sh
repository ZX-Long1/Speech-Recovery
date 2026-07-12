#!/bin/bash
# Full pipeline: whitebox sp4 → blackbox sp1-4 → evaluation
# Per GPU: sequential sp4 then blackbox

declare -A GPU_MAP=( [0]=0 [0.5]=1 [1]=2 [1.5]=3 [2]=5 )
LOG_DIR=/tmp/sc_full_pipeline
mkdir -p "$LOG_DIR"

WB_SCRIPT=/media/sda1/zxlong/tmc_re/sc_whitebox_attack_nTV.py
BB_SCRIPT=/media/sda1/zxlong/tmc_re/sc_blackbox_attack_nTV.py
EVAL_SCRIPT=/media/sda1/zxlong/tmc_re/sc_evaluation_nTV.py

echo "[$(date)] Starting full pipeline..."

for NTV in 0 0.5 1 1.5 2; do
    GPU=${GPU_MAP[$NTV]}
    LOG="$LOG_DIR/full_nTV${NTV}.log"

    # Determine batch size for blackbox based on GPU memory
    # GPU5 = A40 46GB, others = 4090 24GB
    if [ "$GPU" = "5" ]; then
        BB_BS=512
    else
        BB_BS=256
    fi

    nohup bash -c "
        export OMP_NUM_THREADS=2


        # Step 1: Whitebox sp4 (100 epochs)
        # 4090 (GPU0-3): bs=64 due to Conv_sp4 memory; A40 (GPU5): bs=128
        if [ "$GPU" = "5" ]; then WB_BS=128; else WB_BS=64; fi
        echo \"[\$(date)] ===== WHITEBOX SP4 nTV=$NTV GPU=$GPU (bs=\$WB_BS) =====\"
        python '$WB_SCRIPT' --nTV $NTV --cuda_id $GPU --split_point 4 --epochs 100 --batch-size \$WB_BS --log-interval 50 --num_workers 0
        echo \"[\$(date)] ===== WHITEBOX SP4 DONE =====\"

        # Step 2: Blackbox sp1→sp2→sp3→sp4 (50 epochs each, per-SP batch size)
        for SP in 1 2 3 4; do
            # Choose batch size: sp1-2 on 4090 use 512, sp3=256, sp4=128; A40 always 512
            if [ \"\$SP\" = \"4\" ] && [ \"$GPU\" != \"5\" ]; then
                BS=128
            elif [ \"\$SP\" = \"3\" ] && [ \"$GPU\" != \"5\" ]; then
                BS=256
            else
                BS=$BB_BS
            fi
            echo \"[\$(date)] ===== BLACKBOX nTV=$NTV GPU=$GPU sp\$SP (bs=\$BS) =====\"
            python '$BB_SCRIPT' --nTV $NTV --cuda_id $GPU --split_point \$SP --epochs 50 --batch-size \$BS --log-interval 50 --num_workers 0
            echo \"[\$(date)] ===== BLACKBOX sp\$SP DONE =====\"
        done

        # Step 3: Evaluation
        echo \"[\$(date)] ===== EVALUATION nTV=$NTV GPU=$GPU =====\"
        python '$EVAL_SCRIPT' --nTV $NTV --gpu $GPU
        echo \"[\$(date)] ===== EVALUATION DONE =====\"

        echo \"[\$(date)] ===== ALL COMPLETE for nTV=$NTV =====\"
    " > "$LOG" 2>&1 &

    sleep 5
done

echo "[$(date)] All 5 pipelines launched."
echo "Monitor: tail -f $LOG_DIR/full_nTV*.log"
