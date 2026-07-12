#!/usr/bin/env python3
"""Sequential launcher for sp3/sp4 speech_tf training.
Run with: nohup python -u retry_seq.py > /tmp/retry_seq.log 2>&1 &"""
import subprocess, sys, os

LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SCRIPT = '/media/sda1/zxlong/tmc_re/sc_whitebox_attack_nTV.py'
CWD = '/media/sda1/zxlong/tmc_re'
BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'

jobs = [
    (4, 3),  # GPU4, sp3
    (5, 4),  # GPU5, sp4
]

for gpu, sp in jobs:
    for lam in LAMBDAS:
        out_dir = f'{BASE}/Result/whitebox_shallow_speech_tf_nTV{lam}/split_point{sp}/train_record/'
        log = os.path.join(out_dir, 'train.log')
        os.makedirs(out_dir, exist_ok=True)

        cmd = [sys.executable, '-u', SCRIPT,
               f'--split_point={sp}', f'--batch-size=128', f'--cuda_id={gpu}',
               f'--nTV={lam}', f'--regularizer=speech_tf', f'--alpha=1.0']

        print(f'[GPU{gpu} sp{sp}] Starting lam={lam}', flush=True)
        with open(log, 'w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=CWD).returncode
        print(f'[GPU{gpu} sp{sp}] Finished lam={lam} rc={rc}', flush=True)

print('All jobs completed.', flush=True)
