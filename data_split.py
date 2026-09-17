import os, random
from collections import defaultdict

OUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/manifests'
SC_BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'
CREMAD = '/media/sda1/zxlong/tmc_re/CREMA-D'

os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42


def stratified_5to1(files, key_fn, out_prefix, store_full=False):
    groups = defaultdict(list)
    for f in files:
        groups[key_fn(f)].append(f)
    rng = random.Random(SEED)
    aux, train = [], []
    for k in sorted(groups):
        g = groups[k]
        rng.shuffle(g)
        n_aux = max(1, round(len(g) / 6.0))
        aux += g[:n_aux]
        train += g[n_aux:]
    with open(f'{OUT_DIR}/{out_prefix}_aux.txt', 'w') as f:
        f.write('\n'.join(aux) + '\n')
    with open(f'{OUT_DIR}/{out_prefix}_train.txt', 'w') as f:
        f.write('\n'.join(train) + '\n')
    print(f'{out_prefix}: total={len(files)}  train={len(train)}  aux={len(aux)}')
    return len(aux)


# --- SC: split target model training set (train_set_mel_spect_numpy) 5:1 ---
sc_dir = f'{SC_BASE}/audio/train_set_mel_spect_numpy'
sc_files = sorted(os.listdir(sc_dir))
stratified_5to1(sc_files, key_fn=lambda f: f.split('_')[0], out_prefix='sc')

# --- CREMA-D: merge processed/train + processed/aux, then 5:1 ---
cremad_files = []
for split in ['train', 'aux']:
    d = f'{CREMAD}/processed/{split}'
    for f in sorted(os.listdir(d)):
        if f.endswith('_mel.npy'):
            cremad_files.append(f'{split}/{f}')
stratified_5to1(cremad_files, key_fn=lambda f: os.path.basename(f).split('_')[2],
                out_prefix='cremad')

# --- test manifests (unchanged test sets) ---
sc_test_files = sorted(os.listdir(f'{SC_BASE}/audio/test_set_mel_spect_numpy'))
with open(f'{OUT_DIR}/sc_test.txt', 'w') as f:
    f.write('\n'.join(sc_test_files) + '\n')
print(f'sc_test: {len(sc_test_files)}')

cremad_test_files = sorted(f for f in os.listdir(f'{CREMAD}/processed/test')
                           if f.endswith('_mel.npy'))
with open(f'{OUT_DIR}/cremad_test.txt', 'w') as f:
    f.write('\n'.join(f'test/{x}' for x in cremad_test_files) + '\n')
print(f'cremad_test: {len(cremad_test_files)}')

print('done')
