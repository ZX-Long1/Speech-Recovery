#!/usr/bin/env python3
"""Launch speech_tf training per-GPU shell scripts that run tasks sequentially."""

import subprocess, sys, os

REGULARIZER = sys.argv[1] if len(sys.argv) > 1 else 'speech_tf'
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SC_BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'
SCRIPT = 'sc_whitebox_attack_nTV.py'
CWD = os.path.dirname(os.path.abspath(__file__))

jobs = {
    0: [(1, 128, l) for l in [0.0, 1.0, 2.0]],
    1: [(1, 128, l) for l in [0.5, 1.5]],
    2: [(2, 128, l) for l in [0.0, 1.0, 2.0]],
    3: [(2, 128, l) for l in [0.5, 1.5]],
    4: [(3, 128, l) for l in LAMBDAS],
    5: [(4, 128, l) for l in LAMBDAS],
}

for gpu, tasks in jobs.items():
    script_lines = ['#!/bin/bash', 'set -e', '']
    for sp, bs, lam in tasks:
        out_dir = f'{SC_BASE}/Result/whitebox_shallow_{REGULARIZER}_nTV{lam}/split_point{sp}/train_record/'
        log = os.path.join(out_dir, 'train.log')
        cmd = f'cd {CWD} && python -u {SCRIPT} --split_point={sp} --batch-size={bs} --cuda_id={gpu} --nTV={lam} --regularizer={REGULARIZER} --alpha={ALPHA}'
        script_lines.append(f'mkdir -p {out_dir}')
        script_lines.append(f'echo "[GPU{gpu}] Starting sp={sp} lam={lam}"')
        script_lines.append(f'{cmd} > {log} 2>&1')
        script_lines.append(f'echo "[GPU{gpu}] Finished sp={sp} lam={lam} rc=$?"')

    sh_path = f'/tmp/launch_gpu{gpu}.sh'
    with open(sh_path, 'w') as f:
        f.write('\n'.join(script_lines) + '\n')
    os.chmod(sh_path, 0o755)

    subprocess.Popen(['bash', sh_path],
                     stdout=open(f'/tmp/launch_gpu{gpu}.log', 'w'),
                     stderr=subprocess.STDOUT,
                     start_new_session=True,
                     cwd=CWD)
    print(f'Launched GPU{gpu}: {len(tasks)} tasks sequential', flush=True)

print(f'\nAll {sum(len(v) for v in jobs.values())} runs launched across 6 GPUs.')
print('Logs: /tmp/launch_gpu{0,1,2,3,4,5}.log')
