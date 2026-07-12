#!/usr/bin/env python3
"""Evaluate speech_tf models: one eval per lambda, each handles all 4 split points."""

import subprocess, sys, os, time

LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sc_eval_speech_tf.py')

# Assign evals to idle GPUs (0-5, skip 6)
eval_jobs = [
    (0.0, 'none', 0),   # none baseline (reuses nTV=0.0 weights)
    (0.0, 'speech_tf', 1),
    (0.5, 'speech_tf', 2),
    (1.0, 'speech_tf', 3),
    (1.5, 'speech_tf', 4),
    (2.0, 'speech_tf', 5),
]

procs = []
for lam, reg, gpu in eval_jobs:
    cmd = [sys.executable, '-u', SCRIPT,
           f'--gpu={gpu}', f'--nTV={lam}', f'--regularizer={reg}']

    out_dir = f'sc_eval_results_mel_{reg}_nTV{lam}'
    os.makedirs(out_dir, exist_ok=True)
    log = f'{out_dir}/eval.log'

    with open(log, 'w') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                             start_new_session=True, close_fds=True)
    procs.append(p)
    print(f'Launched {reg} nTV={lam} GPU{gpu} (pid={p.pid})')
    time.sleep(10)

print(f'\nAll {len(procs)} evaluations launched.')
print('Monitor: tail -f sc_eval_results_mel_{reg}_nTV{lam}/eval.log')
