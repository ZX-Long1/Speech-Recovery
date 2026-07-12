#!/usr/bin/env python3
"""Retrain sp3/sp4 speech_tf models on GPU4/5."""

import subprocess, os

SCRIPT = '/media/sda1/zxlong/tmc_re/sc_whitebox_attack_nTV.py'
CWD = '/media/sda1/zxlong/tmc_re'
BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'
LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]

for gpu, sp in [(4, 3), (5, 4)]:
    lines = ['#!/bin/bash', 'set -e', '']
    for lam in LAMBDAS:
        out_dir = f'{BASE}/Result/whitebox_shallow_speech_tf_nTV{lam}/split_point{sp}/train_record'
        log = f'{out_dir}/train.log'
        lines.append(f'mkdir -p {out_dir}')
        lines.append(f'echo "[GPU{gpu}] Starting sp={sp} lam={lam}"')
        lines.append(f'cd {CWD} && python -u {SCRIPT} --split_point={sp} --batch-size=128 --cuda_id={gpu} --nTV={lam} --regularizer=speech_tf --alpha=1.0 > {log} 2>&1')
        lines.append(f'echo "[GPU{gpu}] Finished sp={sp} lam={lam} rc=$?"')
        lines.append('')

    sh_path = f'/tmp/retry_gpu{gpu}.sh'
    with open(sh_path, 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(sh_path, 0o755)

    subprocess.Popen(['bash', sh_path],
                     stdout=open(f'/tmp/retry_gpu{gpu}.log', 'w'),
                     stderr=subprocess.STDOUT,
                     start_new_session=True,
                     cwd=CWD)
    print(f'Launched GPU{gpu}: sp={sp} 5 lambdas sequential')
