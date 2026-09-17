import os, csv, re
import random
import argparse as _argparse
from collections import defaultdict

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
OUT_DIR = '/media/sda1/zxlong/tmc_re/speech_inver/results/wer'
TAB_DIR = '/media/sda1/zxlong/tmc_re/speech_inver/tables'

SENTENCE_MAP = {
    'IEO': "it's eleven o'clock",
    'TIE': 'that is exactly what happened',
    'IOM': "i'm on my way to the meeting",
    'IWW': 'i wonder what this is about',
    'TAI': 'the airplane is almost full',
    'MTI': 'maybe tomorrow it will be cold',
    'IWL': 'i would like a new alarm clock',
    'ITH': "i think i have a doctor's appointment",
    'DFA': "don't forget a jacket",
    'ITS': "i think i've seen this before",
    'TSI': 'the surface is slick',
    'WSI': "we'll stop in a couple of minutes",
}

def norm(t):
    t = t.lower().replace("'", "")
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return ' '.join(t.split())

def edist(r, h):
    r, h = r.split(), h.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)]

def code_of(sample):
    return sample.split('_')[1]

def aggregate(rows, keyfn, samplekey, hypkey):
    errs = defaultdict(int)
    words = defaultdict(int)
    per_sample = defaultdict(list)
    items = defaultdict(list)
    for x in rows:
        k = keyfn(x)
        if k is None:
            continue
        gt = SENTENCE_MAP.get(code_of(x[samplekey]))
        if gt is None:
            continue
        e = edist(norm(gt), norm(x[hypkey]))
        n = len(norm(gt).split())
        errs[k] += e
        words[k] += n
        per_sample[k].append(e / max(1, n))
        items[k].append((e, n))
    out = {}
    for k in errs:
        out[k] = {
            'corpus': 100.0 * errs[k] / max(1, words[k]),
            'per_sample': 100.0 * sum(per_sample[k]) / len(per_sample[k]),
            'n': len(per_sample[k]),
            'items': items[k],
        }
    return out

def bootstrap_ci(items, n_boot=1000, seed=42):
    rng = random.Random(seed)
    n = len(items)
    wers = []
    for _ in range(n_boot):
        e = w = 0
        for _ in range(n):
            ei, wi = items[rng.randrange(n)]
            e += ei
            w += wi
        wers.append(100.0 * e / max(1, w))
    wers.sort()
    lo = wers[int(0.025 * (n_boot - 1))]
    hi = wers[int(0.975 * (n_boot - 1))]
    return round(lo, 2), round(hi, 2)

def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    results = []

    _rows200 = read_csv(f'{BASE}/eval_results/wer_detail.csv')
    STEMS200 = set(r['sample'] for r in _rows200 if r['feature'] == 'mel')
    print(f'Fixed WER subset: {len(STEMS200)} utterances', flush=True)

    orig = read_csv(f'{BASE}/whisper_inversion/original_wer.csv')
    oe = ow = 0
    ops = []
    oitems = []
    for x in orig:
        if x['file'] not in STEMS200:
            continue
        gt = SENTENCE_MAP.get(code_of(x['file']))
        e = edist(norm(gt), norm(x['hypothesis']))
        n = len(norm(gt).split())
        oe += e
        ow += n
        ops.append(e / max(1, n))
        oitems.append((e, n))
    oci_lo, oci_hi = bootstrap_ci(oitems)
    results.append({
        'source': 'original', 'mode': '-', 'split_point': '-', 'ntv': '-',
        'corpus_wer_pct': round(100.0 * oe / ow, 2),
        'per_sample_wer_pct': round(100.0 * sum(ops) / len(ops), 2),
        'n': len(ops),
        'n_evaluated': len(ops),
        'n_failures': 0,
        'corpus_ci_low': oci_lo,
        'corpus_ci_high': oci_hi,
    })
    print(f'Original (GT->Whisper(orig)): corpus={100.0*oe/ow:.2f}%  '
          f'95%CI=[{oci_lo}, {oci_hi}]  n={len(ops)}')

    deep = read_csv(f'{BASE}/eval_results/wer_detail.csv')
    deep = [x for x in deep if x['sample'] in STEMS200]
    mel_combos = ([(m, s) for m in ['blackbox', 'whitebox', 'whitebox_shallow']
                   for s in [1, 2, 3, 4] if not (m == 'blackbox' and s == 4)])
    deep_agg = aggregate(
        deep,
        keyfn=lambda x: (x['mode'], int(x['split_point']))
        if x['feature'] == 'mel' and (x['mode'], int(x['split_point'])) in mel_combos
        else None,
        samplekey='sample', hypkey='recon_text',
    )
    for m, s in mel_combos:
        r = deep_agg[(m, s)]
        ci_lo, ci_hi = bootstrap_ci(r['items'])
        results.append({
            'source': 'resnet18_deep', 'mode': m, 'split_point': s, 'ntv': '-',
            'corpus_wer_pct': round(r['corpus'], 2),
            'per_sample_wer_pct': round(r['per_sample'], 2), 'n': r['n'],
            'n_evaluated': r['n'],
            'n_failures': 200 - r['n'],
            'corpus_ci_low': ci_lo,
            'corpus_ci_high': ci_hi,
        })
    print('ResNet18 deep mel (GT->Recon) computed')

    ntv_results = {}
    for ntv in ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']:
        rows = read_csv(f'{BASE}/eval_results_mel_nTV{ntv}/wer_detail.csv')
        rows = [x for x in rows if x['sample'] in STEMS200]
        agg = aggregate(
            rows,
            keyfn=lambda x, ntv=ntv: (x['mode'], int(x['split_point']), ntv)
            if x['condition'] == 'clean' else None,
            samplekey='sample', hypkey='recon_text',
        )
        ntv_results[ntv] = agg
        for m in ['blackbox_shallow', 'whitebox_shallow']:
            for s in [1, 2, 3]:
                r = agg[(m, s, ntv)]
                ci_lo, ci_hi = bootstrap_ci(r['items'])
                results.append({
                    'source': 'resnet18_shallow', 'mode': m, 'split_point': s,
                    'ntv': ntv, 'corpus_wer_pct': round(r['corpus'], 2),
                    'per_sample_wer_pct': round(r['per_sample'], 2), 'n': r['n'],
                    'n_evaluated': r['n'],
                    'n_failures': 200 - r['n'],
                    'corpus_ci_low': ci_lo,
                    'corpus_ci_high': ci_hi,
                })
    print('ResNet18 shallow nTV sweeps (GT->Recon) computed')

    whisper_results = {}
    whisper_dirs = [('whisper', 'whisper_inversion'),
                    ('whisper_nes', 'whisper_inversion_nes_ep100_v7')]
    for source_label, base in whisper_dirs:
        for d in ['sp1_blackbox', 'sp1_whitebox', 'sp2_blackbox', 'sp2_whitebox',
                  'sp4_blackbox', 'sp4_whitebox', 'sp8_blackbox', 'sp8_whitebox']:
            path = f'{BASE}/{base}/{d}/eval_results.csv'
            if not os.path.exists(path):
                continue
            rows = read_csv(path)
            rows = [r for r in rows if r['file'] in STEMS200]
            agg = aggregate(rows, keyfn=lambda x, d=d: d, samplekey='file',
                            hypkey='hypothesis')
            r = agg[d]
            sp = int(d.split('_')[0].replace('sp', ''))
            mode = d.split('_')[1]
            whisper_results[(source_label, mode, sp)] = r
            ci_lo, ci_hi = bootstrap_ci(r['items'])
            results.append({
                'source': source_label, 'mode': mode, 'split_point': sp, 'ntv': '-',
                'corpus_wer_pct': round(r['corpus'], 2),
                'per_sample_wer_pct': round(r['per_sample'], 2), 'n': r['n'],
                'n_evaluated': r['n'],
                'n_failures': len(STEMS200) - r['n'],
                'corpus_ci_low': ci_lo,
                'corpus_ci_high': ci_hi,
            })
    print('Whisper inversion (GT->Recon) computed')

    parser = _argparse.ArgumentParser()
    parser.add_argument('--extra_dirs', nargs='*', default=[],
                        help='Dirs each containing eval_results.csv (file,reference,hypothesis,wer,mse,snr)')
    parser.add_argument('--label', type=str, default='5to1')
    a = parser.parse_args()
    for d in a.extra_dirs:
        if not os.path.exists(f'{d}/eval_results.csv'):
            print(f'[WARN] no eval_results.csv in {d}')
            continue
        rows = read_csv(f'{d}/eval_results.csv')
        agg = aggregate(rows, keyfn=lambda x, d=d: os.path.basename(d),
                        samplekey='file', hypkey='hypothesis')
        key = os.path.basename(d)
        r = agg[key]
        ci_lo, ci_hi = bootstrap_ci(r['items'])
        m = re.search(r'sp(\d+)', key)
        sp = int(m.group(1)) if m else '?'
        results.append({
            'source': a.label, 'mode': key, 'split_point': sp, 'ntv': '-',
            'corpus_wer_pct': round(r['corpus'], 2),
            'per_sample_wer_pct': round(r['per_sample'], 2), 'n': r['n'],
            'n_evaluated': r['n'],
            'n_failures': 0,
            'corpus_ci_low': ci_lo,
            'corpus_ci_high': ci_hi,
        })
        print(f'{a.label} [{key}]: corpus={r["corpus"]:.2f}%  '
              f'95%CI=[{ci_lo}, {ci_hi}]  n={r["n"]}')

    csv_path = f'{OUT_DIR}/cremad_gt_recon_wer.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'source', 'mode', 'split_point', 'ntv',
            'corpus_wer_pct', 'per_sample_wer_pct', 'n',
            'n_evaluated', 'n_failures',
            'corpus_ci_low', 'corpus_ci_high'])
        w.writeheader()
        w.writerows(results)
    print(f'Saved {csv_path}')

    tex_main = f'{TAB_DIR}/cremad_gt_recon_wer_main.tex'
    with open(tex_main, 'w') as f:
        f.write('\\begin{table}[ht]\n')
        f.write('\\centering\n')
        f.write('\\caption{WER (GT $\\to$ Recon, \\%) on CREMA-D --- '
                'normalized (lowercase, punctuation stripped), corpus-level}\n')
        f.write('\\label{tab:cremad_gt_recon_wer}\n')
        f.write('\\begin{tabular}{lrrrrr}\n')
        f.write('\\toprule\n')
        f.write('Model & sp1 & sp2 & sp3 & sp4 & sp8 \\\\\n')
        f.write('\\midrule\n')
        f.write('Original (GT $\\to$ Whisper(orig)) & \\multicolumn{5}{c}{5.49} \\\\\n')
        f.write('\\midrule\n')
        rows_def = [
            ('ResNet18 whitebox\\_shallow', 'whitebox_shallow', [1, 2, 3, 4]),
            ('ResNet18 whitebox', 'whitebox', [1, 2, 3, 4]),
            ('ResNet18 blackbox', 'blackbox', [1, 2, 3]),
            ('Whisper blackbox (NES)', 'whisper_nes:blackbox', [1, 2, 4, 8]),
            ('Whisper whitebox', None, [1, 2, 4, 8]),
        ]
        for label, mode, splits in rows_def:
            cells = []
            for sp in splits:
                if isinstance(mode, str) and ':' in mode:
                    src, m = mode.split(':')
                    r = whisper_results[(src, m, sp)]
                elif mode is None:
                    r = whisper_results[('whisper', label.split()[-1], sp)]
                else:
                    r = deep_agg[(mode, sp)]
                cells.append(f'{r["corpus"]:.2f}')
            while len(cells) < 5:
                cells.append('--')
            f.write(f'{label} & ' + ' & '.join(cells) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    print(f'Saved {tex_main}')

    tex_ntv = f'{TAB_DIR}/cremad_gt_recon_wer_ntv.tex'
    with open(tex_ntv, 'w') as f:
        f.write('\\begin{table}[ht]\n')
        f.write('\\centering\n')
        f.write('\\caption{WER (GT $\\to$ Recon, \\%) of shallow ResNet18 '
                'across nTV --- normalized, corpus-level}\n')
        f.write('\\label{tab:cremad_gt_recon_wer_ntv}\n')
        f.write('\\begin{tabular}{lrrr rrr}\n')
        f.write('\\toprule\n')
        f.write('nTV & \\multicolumn{3}{c}{whitebox\\_shallow} & '
                '\\multicolumn{3}{c}{blackbox\\_shallow} \\\\\n')
        f.write(' & sp1 & sp2 & sp3 & sp1 & sp2 & sp3 \\\\\n')
        f.write('\\midrule\n')
        for ntv in ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']:
            wb = ntv_results[ntv]
            cells = []
            for m in ['whitebox_shallow', 'blackbox_shallow']:
                for s in [1, 2, 3]:
                    cells.append(f'{wb[(m, s, ntv)]["corpus"]:.2f}')
            f.write(f'{ntv} & ' + ' & '.join(cells) + ' \\\\\n')
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    print(f'Saved {tex_ntv}')

if __name__ == '__main__':
    main()
