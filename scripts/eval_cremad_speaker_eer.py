import os, sys, math, warnings, argparse
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
parser.add_argument('--mode', choices=['whitebox_shallow', 'blackbox', 'whisper_blackbox', 'whisper_whitebox'], default='whitebox_shallow')
args = parser.parse_args()

MODE = args.mode
if MODE in ('whisper_blackbox', 'whisper_whitebox'):
    SPLIT_POINTS = [1, 2, 4, 8]
else:
    SPLIT_POINTS = [1, 2, 3] if MODE == 'blackbox' else [1, 2, 3, 4]

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
WAV_DIR = f'{BASE}/AudioWAV'
if MODE.startswith('whisper_'):
    EVAL_DIR = f'{BASE}/eval_results/audio_samples/{MODE}'
else:
    EVAL_DIR = f'{BASE}/eval_results/audio_samples/mel_{MODE}'
OUT_DIR = '/media/sda1/zxlong/tmc_re/results/speaker_leakage'
TAB_DIR = '/media/sda1/zxlong/tmc_re/tables'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGET_SR = 16000

print(f'Device: {DEVICE}')
print(f'Mode: {MODE}')
print(f'Eval dir: {EVAL_DIR}')
print(f'Split points: {SPLIT_POINTS}')

import os, shutil
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

def parse_cremad(fname):
    name = os.path.splitext(fname)[0]
    parts = name.split('_')
    if parts[0] in ('clean', 'recon') and parts[1].startswith('sp'):
        speaker = parts[2]
        sentence = parts[3]
    else:
        speaker = parts[0]
        sentence = parts[1]
    return speaker, sentence

print("Building enrollment prototypes from all AudioWAV...")
enroll_embeddings = {}
enroll_count = {}
all_wav_files = sorted([f for f in os.listdir(WAV_DIR) if f.endswith('.wav')])
for i, fname in enumerate(all_wav_files):
    spk, sen = parse_cremad(fname)
    path = os.path.join(WAV_DIR, fname)
    audio, sr = load_audio(path)
    emb = extract_embedding(audio, sr)
    if spk not in enroll_embeddings:
        enroll_embeddings[spk] = []
        enroll_count[spk] = 0
    enroll_embeddings[spk].append(emb)
    enroll_count[spk] += 1
    if (i + 1) % 1000 == 0:
        print(f'  Processed {i+1}/{len(all_wav_files)} enrollment wavs')

print(f'Enrollment done: {len(enroll_embeddings)} speakers, '
      f'{sum(len(v) for v in enroll_embeddings.values())} total utterances')
for spk, embs in sorted(enroll_embeddings.items()):
    enroll_embeddings[spk] = np.mean(embs, axis=0)

def compute_eer(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2.0 * 100
    return eer

def evaluate_condition(condition_label, query_wav_dir, query_files, enroll_embeddings):
    all_scores = []
    all_labels = []
    skipped = 0
    for qfname in query_files:
        spk, sen = parse_cremad(qfname)
        path = os.path.join(query_wav_dir, qfname)
        if not os.path.exists(path):
            skipped += 1
            continue
        audio, sr = load_audio(path)
        q_emb = extract_embedding(audio, sr)
        if spk not in enroll_embeddings:
            skipped += 1
            continue
        pos_score = float(np.dot(q_emb, enroll_embeddings[spk]) /
                          (np.linalg.norm(q_emb) * np.linalg.norm(enroll_embeddings[spk])))
        all_scores.append(pos_score)
        all_labels.append(1)
        for other_spk in enroll_embeddings:
            if other_spk == spk:
                continue
            neg_score = float(np.dot(q_emb, enroll_embeddings[other_spk]) /
                              (np.linalg.norm(q_emb) * np.linalg.norm(enroll_embeddings[other_spk])))
            all_scores.append(neg_score)
            all_labels.append(0)
    eer = compute_eer(np.array(all_scores), np.array(all_labels))
    n_queries = len(query_files) - skipped
    print(f'  {condition_label:20s}: EER={eer:.2f}%  (queries={n_queries}, '
          f'trials={len(all_scores)})')
    return eer, n_queries, len(all_scores)

clean_files_s1 = sorted([f for f in os.listdir(EVAL_DIR) if f.startswith('clean_sp1_')])
recon_files_by_sp = {}
for sp in SPLIT_POINTS:
    recon_files_by_sp[sp] = sorted([f for f in os.listdir(EVAL_DIR)
                                     if f.startswith(f'recon_sp{sp}_')])

print(f'\nQuery samples per split: {len(clean_files_s1)}')
print(f'Recon files per split: { {sp: len(recon_files_by_sp[sp]) for sp in SPLIT_POINTS} }')

results = []
eer, nq, nt = evaluate_condition('Original', EVAL_DIR, clean_files_s1, enroll_embeddings)
results.append({'condition': 'Original', 'eer': round(eer, 2), 'queries': nq, 'trials': nt})

for sp in SPLIT_POINTS:
    eer, nq, nt = evaluate_condition(f'Recon_Sp{sp}', EVAL_DIR,
                                      recon_files_by_sp[sp], enroll_embeddings)
    results.append({'condition': f'Recon_Sp{sp}', 'eer': round(eer, 2),
                    'queries': nq, 'trials': nt})

df = pd.DataFrame(results)
df['eer'] = df['eer'].apply(lambda x: f'{x:.2f}')
csv_path = f'{OUT_DIR}/cremad_ecapa_eer_{MODE}.csv'
df.to_csv(csv_path, index=False)
print(f'\nSaved: {csv_path}')
print(df.to_string(index=False))

# LaTeX table (per mode)
tex_path = f'{TAB_DIR}/cremad_speaker_eer_{MODE}.tex'
with open(tex_path, 'w') as f:
    f.write('\\begin{table}[ht]\n')
    f.write('\\centering\n')
    f.write(f'\\caption{{Speaker Verification EER (\\%) on CREMA-D — ECAPA-TDNN ({MODE})}}\n')
    f.write(f'\\label{{tab:cremad_speaker_eer_{MODE}}}\n')
    f.write('\\begin{tabular}{lrr}\n')
    f.write('\\toprule\n')
    f.write('Condition & EER (\\%) & Trials \\\\\n')
    f.write('\\midrule\n')
    for r in results:
        cond = r['condition'].replace('_', ' ')
        esc = ' \\\\'
        f.write(f'{cond} & {r["eer"]:.2f} & {r["trials"]}{esc}\n')
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}\n')
    f.write('\\end{table}\n')
print(f'Saved: {tex_path}')

# README
query_speakers = len(set(f.split('_')[2] for f in clean_files_s1))
readme_path = f'{OUT_DIR}/README.md'
with open(readme_path, 'w') as f:
    f.write('# Speaker Verification EER on CREMA-D (ECAPA-TDNN)\n\n')
    f.write('## Data\n')
    f.write(f'- **Enrollment**: `{WAV_DIR}` — ')
    f.write(f'{len(all_wav_files)} utterances, {len(enroll_embeddings)} speakers\n')
    f.write(f'- **Query (Original + Recon Sp1-4)**: `{EVAL_DIR}` — ')
    f.write(f'{len(clean_files_s1)} utterances per condition, ')
    f.write(f'{query_speakers} speakers\n\n')
    f.write('## Protocol\n')
    f.write('- ECAPA-TDNN pretrained on VoxCeleb (`speechbrain/spkrec-ecapa-voxceleb`)\n')
    f.write('- All audio resampled to 16 kHz mono\n')
    f.write('- Enrollment: mean embedding of all original utterances per speaker\n')
    f.write('- Query: each utterance excluded from its own enrollment\n')
    f.write('- Scoring: cosine similarity\n')
    f.write('- EER computed from continuous scores over all positive/negative trials\n\n')
    f.write('## Files\n')
    f.write(f'- `cremad_ecapa_eer.csv`: condition-level EER summary\n')
    f.write(f'- `../../scripts/eval_cremad_speaker_eer.py`: evaluation script\n\n')
    f.write('## Run\n')
    f.write('```bash\n')
    f.write(f'python ../../scripts/eval_cremad_speaker_eer.py --mode {MODE}\n')
    f.write('```\n')
print(f'Saved: {readme_path}')

print('\nAll done.')
