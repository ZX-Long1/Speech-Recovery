import os, sys, math, warnings, argparse
from collections import Counter
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve

import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['whitebox_shallow', 'blackbox', 'whisper_blackbox', 'whisper_whitebox', 'whisper_nes_blackbox'], default='whitebox_shallow')
parser.add_argument('--n_boot', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--suffix', type=str, default='loo')
parser.add_argument('--stems_file', type=str, default='',
                    help='Optional file listing query stems to restrict evaluation')
parser.add_argument('--eval_dir', type=str, default='',
                    help='Override audio dir containing clean_sp1_*/recon_sp*_*.wav')
parser.add_argument('--out_dir', type=str,
                    default='/media/sda1/zxlong/tmc_re/speech_inver/results/eer',
                    help='Output dir for CSVs (default: speech_inver/results/eer)')
parser.add_argument('--tab_dir', type=str,
                    default='/media/sda1/zxlong/tmc_re/speech_inver/tables',
                    help='Output dir for tex tables')
parser.add_argument('--split_points', type=int, nargs='+', default=[],
                    help='Override SPLIT_POINTS (e.g. 1 3)')
args = parser.parse_args()

MODE = args.mode
if MODE in ('whisper_blackbox', 'whisper_whitebox', 'whisper_nes_blackbox'):
    SPLIT_POINTS = [1, 2, 4, 8]
else:
    SPLIT_POINTS = [1, 2, 3] if MODE == 'blackbox' else [1, 2, 3, 4]
if args.split_points:
    SPLIT_POINTS = args.split_points

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
WAV_DIR = f'{BASE}/AudioWAV'
if args.eval_dir:
    EVAL_DIR = args.eval_dir
elif MODE.startswith('whisper_'):
    EVAL_DIR = f'{BASE}/eval_results/audio_samples/{MODE}'
else:
    EVAL_DIR = f'{BASE}/eval_results/audio_samples/mel_{MODE}'
OUT_DIR = args.out_dir
TAB_DIR = args.tab_dir
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SR = 16000

print(f'Device: {DEVICE}')
print(f'Mode: {MODE}')
print(f'Eval dir: {EVAL_DIR}')
print(f'Split points: {SPLIT_POINTS}')
print(f'Bootstrap: n_boot={args.n_boot} seed={args.seed}')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HUGGINGFACE_HUB_OFFLINE'] = '1'
MODEL_DIR = '/media/sda1/zxlong/tmc_re/model/spkrec-ecapa-voxceleb'
speaker_encoder = EncoderClassifier.from_hparams(
    source=MODEL_DIR,
    savedir=MODEL_DIR,
    hparams_file='hyperparams_local.yaml',
    run_opts={'device': str(DEVICE)},
)

def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    if sr != TARGET_SR:
        import scipy.signal
        audio = scipy.signal.resample(audio, int(len(audio) * TARGET_SR / sr))
        sr = TARGET_SR
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    return audio, sr

def extract_embedding(audio, sr=TARGET_SR):
    with torch.no_grad():
        waveform = torch.from_numpy(audio).float().to(DEVICE).unsqueeze(0)
        emb = speaker_encoder.encode_batch(waveform, wav_lens=torch.tensor([1.0]).to(DEVICE))
        return emb.squeeze(0).squeeze(0).cpu().numpy()

def parse_query(fname):
    name = os.path.splitext(fname)[0]
    parts = name.split('_')
    if parts[0] in ('clean', 'recon') and parts[1].startswith('sp'):
        speaker = parts[2]
        source_stem = '_'.join(parts[2:])
    else:
        speaker = parts[0]
        source_stem = name
    return speaker, source_stem

restrict_stems = None
if args.stems_file:
    with open(args.stems_file) as f:
        restrict_stems = set(l.strip() for l in f if l.strip())
    print(f'Restricting queries to {len(restrict_stems)} stems from {args.stems_file}')

def in_query_set(fname):
    if restrict_stems is None:
        return True
    _, stem = parse_query(fname)
    return stem in restrict_stems

print("Building per-utterance enrollment embeddings from all original AudioWAV...")
enroll_emb = {}
enroll_sum = {}
enroll_count = {}
all_wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith('.wav')])
for i, fname in enumerate(all_wav_files):
    stem = os.path.splitext(fname)[0]
    spk = stem.split('_')[0]
    path = os.path.join(WAV_DIR, fname)
    audio, sr = load_audio(path)
    emb = extract_embedding(audio, sr)
    enroll_emb.setdefault(spk, {})[stem] = emb
    if spk not in enroll_sum:
        enroll_sum[spk] = np.zeros_like(emb)
        enroll_count[spk] = 0
    enroll_sum[spk] += emb
    enroll_count[spk] += 1
    if (i + 1) % 1000 == 0:
        print(f'  Processed {i+1}/{len(all_wav_files)} enrollment wavs', flush=True)

n_speakers = len(enroll_emb)
full_proto = {spk: enroll_sum[spk] / enroll_count[spk] for spk in enroll_sum}
print(f'Enrollment done: {n_speakers} speakers, '
      f'{sum(len(v) for v in enroll_emb.values())} original utterances', flush=True)

def target_prototype(spk, exclude_stem):
    if spk not in enroll_emb:
        return None, 'speaker_not_enrolled'
    if exclude_stem not in enroll_emb[spk]:
        return None, 'source_not_in_enrollment'
    if enroll_count[spk] <= 1:
        return None, 'no_other_enrollment'
    proto = (enroll_sum[spk] - enroll_emb[spk][exclude_stem]) / (enroll_count[spk] - 1)
    return proto, None

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0 * 100
    return eer

def bootstrap_ci(query_scores, query_labels, n_boot, seed):
    rng = np.random.RandomState(seed)
    n = len(query_scores)
    eers = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        sc = np.concatenate([query_scores[i] for i in idx])
        lb = np.concatenate([query_labels[i] for i in idx])
        eers.append(compute_eer(sc, lb))
    lo, hi = np.percentile(eers, [2.5, 97.5])
    return lo, hi

def evaluate_condition(condition_label, query_wav_dir, query_files):
    query_scores = []
    query_labels = []
    skipped = []
    n_evaluated = 0
    for qfname in query_files:
        path = os.path.join(query_wav_dir, qfname)
        if not os.path.exists(path):
            skipped.append((qfname, 'missing_wav'))
            continue
        spk, source_stem = parse_query(qfname)
        proto, reason = target_prototype(spk, source_stem)
        if reason is not None:
            skipped.append((qfname, reason))
            continue
        audio, sr = load_audio(path)
        q_emb = extract_embedding(audio, sr)
        pos_score = cosine(q_emb, proto)
        block_scores = [pos_score]
        block_labels = [1]
        for other_spk in full_proto:
            if other_spk == spk:
                continue
            neg_score = cosine(q_emb, full_proto[other_spk])
            block_scores.append(neg_score)
            block_labels.append(0)
        query_scores.append(np.array(block_scores))
        query_labels.append(np.array(block_labels))
        n_evaluated += 1

    all_scores = np.concatenate(query_scores)
    all_labels = np.concatenate(query_labels)
    n_genuine = int(all_labels.sum())
    n_impostor = int(len(all_labels) - n_genuine)
    eer = compute_eer(all_scores, all_labels) if len(query_scores) > 0 else float('nan')
    ci_lo, ci_hi = bootstrap_ci(query_scores, query_labels, args.n_boot, args.seed) if len(query_scores) > 0 else (float('nan'), float('nan'))
    reasons = Counter(r for _, r in skipped)
    reason_str = '; '.join(f'{k}:{v}' for k, v in sorted(reasons.items()))
    print(f'  {condition_label:20s}: EER={eer:.2f}%  95%CI=[{ci_lo:.2f}, {ci_hi:.2f}]  '
          f'(speakers={n_speakers}, queries={len(query_files)}, evaluated={n_evaluated}, '
          f'genuine={n_genuine}, impostor={n_impostor}, skipped={len(skipped)} [{reason_str}])', flush=True)
    return {
        'condition': condition_label,
        'n_speakers': n_speakers,
        'n_queries': len(query_files),
        'n_evaluated': n_evaluated,
        'n_genuine': n_genuine,
        'n_impostor': n_impostor,
        'n_skipped': len(skipped),
        'skip_reasons': reason_str,
        'eer_pct': round(eer, 2) if eer == eer else '',
        'eer_ci_low': round(ci_lo, 2) if ci_lo == ci_lo else '',
        'eer_ci_high': round(ci_hi, 2) if ci_hi == ci_hi else '',
    }

clean_files_s1 = sorted([f for f in os.listdir(EVAL_DIR)
                          if f.startswith('clean_sp1_') and in_query_set(f)])
recon_files_by_sp = {}
for sp in SPLIT_POINTS:
    recon_files_by_sp[sp] = sorted([f for f in os.listdir(EVAL_DIR)
                                     if f.startswith(f'recon_sp{sp}_') and in_query_set(f)])

print(f'\nQuery samples per split: {len(clean_files_s1)}')
print(f'Recon files per split: { {sp: len(recon_files_by_sp[sp]) for sp in SPLIT_POINTS} }')

results = []
results.append(evaluate_condition('Original', EVAL_DIR, clean_files_s1))
for sp in SPLIT_POINTS:
    results.append(evaluate_condition(f'Recon_Sp{sp}', EVAL_DIR, recon_files_by_sp[sp]))

df = pd.DataFrame(results)
csv_path = f'{OUT_DIR}/cremad_ecapa_eer_{MODE}_{args.suffix}.csv'
df.to_csv(csv_path, index=False)
print(f'\nSaved: {csv_path}')
print(df.to_string(index=False))

tex_path = f'{TAB_DIR}/cremad_speaker_eer_{MODE}_{args.suffix}.tex'
with open(tex_path, 'w') as f:
    f.write('\\begin{table}[ht]\n')
    f.write('\\centering\n')
    f.write(f'\\caption{{Speaker Verification EER (\\%) on CREMA-D --- ECAPA-TDNN '
            f'({MODE}), strict leave-one-out}}\n')
    f.write(f'\\label{{tab:cremad_speaker_eer_{MODE}_{args.suffix}}}\n')
    f.write('\\begin{tabular}{lrrrr}\n')
    f.write('\\toprule\n')
    f.write('Condition & EER (\\%) & 95\\% CI & Genuine & Impostor \\\\\n')
    f.write('\\midrule\n')
    for r in results:
        cond = r['condition'].replace('_', ' ')
        f.write(f"{cond} & {r['eer_pct']} & [{r['eer_ci_low']}, {r['eer_ci_high']}] & "
                f"{r['n_genuine']} & {r['n_impostor']} \\\\\n")
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}\n')
    f.write('\\end{table}\n')
print(f'Saved: {tex_path}')

print('\nAll done.')
