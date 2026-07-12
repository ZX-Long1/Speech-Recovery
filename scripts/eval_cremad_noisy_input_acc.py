import os, sys, random, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cremad_experiment.cremad_models import BasicBlock, ResNet18

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
WAV_DIR = f'{BASE}/AudioWAV'
CLS_PATH = f'{BASE}/classifier/mel_train_record/classifier.pth'
NOISE_DIR = '/media/sda1/zxlong/tmc_re/noise'
EVAL_DIR = f'{BASE}/eval_results/audio_samples/mel_whitebox_shallow'
PROCESSED = f'{BASE}/processed'
OUT_DIR = '/media/sda1/zxlong/tmc_re/results/speaker_leakage'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_RATE = 16000
N_MELS = 64
MAX_TIME = 640
N_FFT = 320
HOP_LENGTH = 80

print(f'Device: {DEVICE}')

norm_info = json.load(open(f'{PROCESSED}/norm_info.json'))
MEL_MIN, MEL_MAX = norm_info['mel_min'], norm_info['mel_max']

EMO_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
EMO_INV = {v: k for k, v in EMO_MAP.items()}

def feat_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] > MAX_TIME:
        return mel_db[:, :MAX_TIME]
    import scipy.interpolate as sci
    h, w = mel_db.shape
    padded = np.zeros((h, MAX_TIME))
    padded[:, :w] = mel_db
    for i in range(h):
        x_new = np.linspace(0, w - 1, MAX_TIME)
        x_old = np.arange(w)
        fi = sci.interp1d(x_old, mel_db[i], kind='linear', fill_value='extrapolate')
        padded[i] = fi(x_new)
    return padded

def normalize(feat_db):
    norm = ((feat_db - MEL_MIN) / (MEL_MAX - MEL_MIN)) * 255
    return torch.from_numpy(np.uint8(norm)).unsqueeze(0).unsqueeze(0).float() / 255.0

def add_noise(audio, noise, amp=1.0):
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
    noise = noise[:len(audio)]
    audio_rms = np.sqrt(np.mean(audio ** 2))
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-8
    scaled_noise = noise * (audio_rms / noise_rms) * amp
    noisy = audio + scaled_noise
    peak = np.max(np.abs(noisy))
    if peak > 0:
        noisy = noisy / peak
    return noisy

print('Loading classifier...')
classifier = ResNet18(BasicBlock, 1, 6).to(DEVICE)
ckpt = torch.load(CLS_PATH, map_location='cpu')
classifier.load_state_dict(ckpt['model'])
classifier.eval()
print('Classifier loaded.')

clean_files = sorted([f for f in os.listdir(EVAL_DIR) if f.startswith('clean_sp1_')])
audio_wav_files = set(os.listdir(WAV_DIR))

def find_wav_path(spk, sen, emo):
    prefix = f'{spk}_{sen}_{emo}_'
    for fname in audio_wav_files:
        if fname.startswith(prefix) and fname.endswith('.wav'):
            return os.path.join(WAV_DIR, fname)
    return None

test_samples = []
for fname in clean_files:
    parts = fname.rstrip('.wav').split('_')
    spk = parts[2]
    sen = parts[3]
    emo_code = parts[4]
    label = EMO_MAP.get(emo_code, -1)
    wav_path = find_wav_path(spk, sen, emo_code)
    if wav_path is None:
        print(f'WARNING: no wav found for {spk}_{sen}_{emo_code}, skipping')
        continue
    test_samples.append({
        'fname': os.path.basename(wav_path),
        'spk': spk,
        'sen': sen,
        'emo': emo_code,
        'label': label,
        'wav_path': wav_path,
    })
print(f'Test samples: {len(test_samples)}')

noise_types = ['market', 'office', 'street']
noise_cache = {}
for nt in noise_types:
    path = f'{NOISE_DIR}/{nt}.wav'
    noise_audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    noise_cache[nt] = noise_audio

results = []
for sample in tqdm(test_samples, desc='Evaluating noisy input accuracy'):
    audio, _ = librosa.load(sample['wav_path'], sr=SAMPLE_RATE, mono=True)
    label = sample['label']
    for nt in noise_types:
        noise_audio = noise_cache[nt]
        noisy_audio = add_noise(audio, noise_audio, amp=1.0)
        feat_db = feat_mel(noisy_audio)
        inp = normalize(feat_db).to(DEVICE)
        with torch.no_grad():
            logit = classifier(inp, split_point=0)
        correct = int(logit.argmax(dim=1).item() == label)
        results.append({'noise': nt, 'sample': sample['fname'], 'correct': correct})

df = pd.DataFrame(results)

print('\n=== Noisy Input Direct Accuracy ===')
summary = df.groupby('noise')['correct'].agg(['mean', 'count'])
summary['mean'] = summary['mean'] * 100
print(summary.to_string())

# Compare with existing noise reconstruction results
comp_dir = f'{BASE}/eval_results_noise'
print('\n=== Comparison: Noisy Input vs Reconstructed Noisy ===')
modes = ['whitebox_shallow', 'blackbox']
comp_rows = []
for nt in noise_types:
    path = f'{comp_dir}/{nt}/eval_summary.csv'
    if not os.path.exists(path):
        print(f'  WARNING: {path} not found')
        continue
    df_noise = pd.read_csv(path)
    noisy_input_acc = summary.loc[nt, 'mean']
    for mode in modes:
        sub = df_noise[(df_noise['feature'] == 'mel') & (df_noise['mode'] == mode)]
        for _, row in sub.iterrows():
            sp = row['split_point']
            recon_noisy_acc = row['acc_recon_noisy']
            recon_noisy_drop = row['acc_drop_noisy']
            input_drop = 60.46511627906976 - noisy_input_acc  # acc_input for mel
            comp_rows.append({
                'noise': nt,
                'mode': mode,
                'split_point': sp,
                'acc_noisy_input': round(noisy_input_acc, 2),
                'acc_recon_noisy': round(recon_noisy_acc, 2),
                'acc_input_clean': round(60.47, 2),
                'drop_noisy_input': round(input_drop, 2),
                'drop_recon_noisy': round(recon_noisy_drop, 2),
                'diff_acc': round(noisy_input_acc - recon_noisy_acc, 2),
                'diff_drop': round(input_drop - recon_noisy_drop, 2),
            })

df_comp = pd.DataFrame(comp_rows)
csv_path = f'{OUT_DIR}/cremad_noisy_input_vs_recon.csv'
df_comp.to_csv(csv_path, index=False)
print(df_comp.to_string(index=False))
print(f'\nSaved: {csv_path}')

# Summary by noise (aggregated over splits)
print('\n=== Per-Noise Summary (mean over splits) ===')
for mode in modes:
    print(f'\n  MODE: {mode}')
    for nt in noise_types:
        sub = df_comp[(df_comp['noise'] == nt) & (df_comp['mode'] == mode)]
        if len(sub) == 0:
            continue
        print(f'\n  {nt.upper()}:')
        print(f'    Noisy input accuracy (mean):  {sub["acc_noisy_input"].mean():.2f}%')
        print(f'    Recon noisy accuracy (mean):  {sub["acc_recon_noisy"].mean():.2f}%')
        print(f'    Noisy input drop (mean):      {sub["drop_noisy_input"].mean():.2f}%')
        print(f'    Recon noisy drop (mean):      {sub["drop_recon_noisy"].mean():.2f}%')
        print(f'    Accuracy difference:           {sub["diff_acc"].mean():.2f}%')
        print(f'    Drop difference:               {sub["diff_drop"].mean():.2f}%')

print('\nAll done.')
