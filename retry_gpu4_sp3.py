#!/usr/bin/env python3
"""Sequential launcher for sp3 speech_tf training on GPU4."""
import subprocess, sys, os, signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SCRIPT = '/media/sda1/zxlong/tmc_re/sc_whitebox_attack_nTV.py'
BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'
GPU, SP = 4, 3

for lam in LAMBDAS:
    out_dir = f'{BASE}/Result/whitebox_shallow_speech_tf_nTV{lam}/split_point{SP}/train_record/'
    log = os.path.join(out_dir, 'train.log')
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, '-u', SCRIPT,
           f'--split_point={SP}', '--batch-size=128',
           f'--cuda_id={GPU}', f'--nTV={lam}',
           '--regularizer=speech_tf', '--alpha=1.0']
    print(f'[GPU{GPU} sp{SP}] Starting lam={lam}', flush=True)
    with open(log, 'w') as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                           cwd=os.path.dirname(SCRIPT)).returncode
    print(f'[GPU{GPU} sp{SP}] Finished lam={lam} rc={rc}', flush=True)
print('Done.', flush=True)
