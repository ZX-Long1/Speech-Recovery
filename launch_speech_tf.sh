#!/bin/bash
# Launch speech_tf training - one script per GPU for sequential execution on each

REG=${1:-speech_tf}
ALPHA=${2:-1.0}
LAMBDAS=(0.0 0.5 1.0 1.5 2.0)
BASE="/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command/Result"
DIR="$(dirname "$0")"

echo "Regularizer: $REG, alpha=$ALPHA"
echo "Launching 20 runs across 6 GPUs..."

# === GPU 0: sp1 lam=0.0,1.0,2.0 ===
nohup bash -c '
for lam in 0.0 1.0 2.0; do
    OUTDIR='"'${BASE}/whitebox_shallow_${REG}_nTV${lam}/split_point1/train_record'"'
    mkdir -p "$OUTDIR"
    echo "GPU0: sp=1 lam=$lam bs=128"
    python -u '"${DIR}"'/sc_whitebox_attack_nTV.py --split_point=1 --batch-size=128 --cuda_id=0 --nTV=$lam --regularizer='"'${REG}'"' --alpha='"'${ALPHA}"' > "$OUTDIR/train.log" 2>&1
done
echo "GPU0 done"
' > /tmp/gpu0.log 2>&1 &

# === GPU 1: sp1 lam=0.5,1.5 ===
nohup bash -c '
for lam in 0.5 1.5; do
    OUTDIR='"'${BASE}/whitebox_shallow_${REG}_nTV${lam}/split_point1/train_record'"'
    mkdir -p "$OUTDIR"
    echo "GPU1: sp=1 lam=$lam bs=128"
    python -u '"${DIR}"'/sc_whitebox_attack_nTV.py --split_point=1 --batch-size=128 --cuda_id=1 --nTV=$lam --regularizer='"'${REG}'"' --alpha='"'${ALPHA}"' > "$OUTDIR/train.log" 2>&1
done
echo "GPU1 done"
' > /tmp/gpu1.log 2>&1 &

# === GPU 2: sp2 lam=0.0,1.0,2.0 ===
nohup bash -c '
for lam in 0.0 1.0 2.0; do
    OUTDIR='"'${BASE}/whitebox_shallow_${REG}_nTV${lam}/split_point2/train_record'"'
    mkdir -p "$OUTDIR"
    echo "GPU2: sp=2 lam=$lam bs=128"
    python -u '"${DIR}"'/sc_whitebox_attack_nTV.py --split_point=2 --batch-size=128 --cuda_id=2 --nTV=$lam --regularizer='"'${REG}'"' --alpha='"'${ALPHA}"' > "$OUTDIR/train.log" 2>&1
done
echo "GPU2 done"
' > /tmp/gpu2.log 2>&1 &

# === GPU 3: sp2 lam=0.5,1.5 ===
nohup bash -c '
for lam in 0.5 1.5; do
    OUTDIR='"'${BASE}/whitebox_shallow_${REG}_nTV${lam}/split_point2/train_record'"'
    mkdir -p "$OUTDIR"
    echo "GPU3: sp=2 lam=$lam bs=128"
    python -u '"${DIR}"'/sc_whitebox_attack_nTV.py --split_point=2 --batch-size=128 --cuda_id=3 --nTV=$lam --regularizer='"'${REG}'"' --alpha='"'${ALPHA}"' > "$OUTDIR/train.log" 2>&1
done
echo "GPU3 done"
' > /tmp/gpu3.log 2>&1 &

# === GPU 4 (A40): sp3 lam=0.0..2.0 bs=128 ===
nohup bash -c '
for lam in 0.0 0.5 1.0 1.5 2.0; do
    OUTDIR='"'${BASE}/whitebox_shallow_${REG}_nTV${lam}/split_point3/train_record'"'
    mkdir -p "$OUTDIR"
    echo "GPU4: sp=3 lam=$lam bs=128"
    python -u '"${DIR}"'/sc_whitebox_attack_nTV.py --split_point=3 --batch-size=128 --cuda_id=4 --nTV=$lam --regularizer='"'${REG}'"' --alpha='"'${ALPHA}"' > "$OUTDIR/train.log" 2>&1
done
echo "GPU4 done"
' > /tmp/gpu4.log 2>&1 &

# === GPU 5 (A40): sp4 lam=0.0..2.0 bs=128 ===
nohup bash -c '
for lam in 0.0 0.5 1.0 1.5 2.0; do
    OUTDIR='"'${BASE}/whitebox_shallow_${REG}_nTV${lam}/split_point4/train_record'"'
    mkdir -p "$OUTDIR"
    echo "GPU5: sp=4 lam=$lam bs=128"
    python -u '"${DIR}"'/sc_whitebox_attack_nTV.py --split_point=4 --batch-size=128 --cuda_id=5 --nTV=$lam --regularizer='"'${REG}'"' --alpha='"'${ALPHA}"' > "$OUTDIR/train.log" 2>&1
done
echo "GPU5 done"
' > /tmp/gpu5.log 2>&1 &

echo "All 6 GPU scripts launched in background."
echo "Monitor with: tail -f /tmp/gpu{0,1,2,3,4,5}.log"
echo "GPU usage: watch -n 30 nvidia-smi"
