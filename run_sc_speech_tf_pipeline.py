import subprocess, sys, time, os, argparse

SC_BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'

LAMBDA_VALS = [0.0, 0.5, 1.0, 1.5, 2.0]
SPLIT_POINTS = [1, 2, 3, 4]

GPU_MEM = {}
import subprocess as sp
r = sp.run(['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader'],
           capture_output=True, text=True)
for line in r.stdout.strip().split('\n'):
    idx, mem = line.split(', ')
    GPU_MEM[int(idx)] = int(mem.replace(' MiB', ''))

def get_idle_gpus():
    r = sp.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader'],
               capture_output=True, text=True)
    idle = []
    for line in r.stdout.strip().split('\n'):
        idx, used = line.split(', ')
        used_mib = int(used.replace(' MiB', ''))
        if used_mib < 100:
            idle.append(int(idx))
    return idle

def get_batch_size(sp, gpu_id):
    mem = GPU_MEM.get(gpu_id, 24564)
    if sp == 4 and mem <= 25000:
        return 64
    return 128

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--regularizer', type=str, default='speech_tf', choices=['tvd', 'speech_tf'])
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()

    print(f'Starting training: reg={args.regularizer}, alpha={args.alpha}')
    print(f'Grid: {len(LAMBDA_VALS)} lambdas × {len(SPLIT_POINTS)} split points = '
          f'{len(LAMBDA_VALS)*len(SPLIT_POINTS)} runs')

    procs = []
    pending = [(lam, sp) for lam in LAMBDA_VALS for sp in SPLIT_POINTS]
    running = []

    print(f'Pending runs: {len(pending)}', flush=True)

    while pending or running:
        idle = get_idle_gpus()
        idle_set = set(idle)
        used_gpus = [r['gpu'] for r in running]
        free_gpus = sorted(idle_set - set(used_gpus))

        if free_gpus and pending:
            gpu = free_gpus[0]
            lam, sp = pending.pop(0)
            bs = get_batch_size(sp, gpu)

            cmd = ['python', 'sc_whitebox_attack_nTV.py',
                   f'--split_point={sp}',
                   f'--batch-size={bs}',
                   f'--cuda_id={gpu}',
                   f'--nTV={lam}',
                   f'--regularizer={args.regularizer}',
                   f'--alpha={args.alpha}']

            out_dir = f'{SC_BASE}/Result/whitebox_shallow_{args.regularizer}_nTV{lam}/split_point{sp}/'
            os.makedirs(f'{out_dir}train_record/', exist_ok=True)
            log_file = open(f'{out_dir}train_record/train.log', 'w')

            p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
            procs.append(p)
            running.append({'proc': p, 'gpu': gpu, 'lam': lam, 'sp': sp, 'log': log_file})
            print(f'Launched GPU{gpu}: reg={args.regularizer} lam={lam} sp={sp} bs={bs} (pid={p.pid})', flush=True)
            continue

        time.sleep(30)
        still_running = []
        for r in running:
            rc = r['proc'].poll()
            if rc is None:
                still_running.append(r)
            else:
                r['log'].close()
                print(f'Finished GPU{r["gpu"]}: reg={args.regularizer} lam={r["lam"]} sp={r["sp"]} (rc={rc})', flush=True)
        running = still_running

    for p in procs:
        p.wait()

    print(f'\nAll {len(procs)} {args.regularizer} runs completed.', flush=True)

if __name__ == '__main__':
    main()
