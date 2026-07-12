import os, sys, random, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import jiwer
import whisper
import pandas as pd
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cremad_experiment.cremad_models import (
    BasicBlock, ResNet18,
    Conv_sp1, Conv_sp2, Conv_sp3, Conv_sp4,
    Conv_sp1_mfcc, Conv_sp2_mfcc, Conv_sp3_mfcc, Conv_sp4_mfcc,
    Conv_sp1_shallow, Conv_sp2_shallow, Conv_sp3_shallow, Conv_sp4_shallow,
)

SAMPLE_RATE = 16000
N_MELS = 64
N_MFCC = 64
MAX_TIME = 640
N_FFT = 320
HOP_LENGTH = 80

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
PROCESSED = f'{BASE}/processed'
WAV_DIR = f'{BASE}/AudioWAV'
OUTPUT_DIR = f'{BASE}/eval_results'
WER_SUBSET = 200
AUDIO_SAMPLES = 20  # per model

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'Device: {DEVICE}', flush=True)

norm_info = json.load(open(f'{PROCESSED}/norm_info.json'))
MEL_MIN, MEL_MAX = norm_info['mel_min'], norm_info['mel_max']
MFCC_MIN, MFCC_MAX = norm_info['mfcc_min'], norm_info['mfcc_max']

INV_CLS_MAP = {
    ('mel', 'whitebox'):         {1: Conv_sp1, 2: Conv_sp2, 3: Conv_sp3, 4: Conv_sp4},
    ('mel', 'whitebox_shallow'): {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow, 4: Conv_sp4_shallow},
    ('mel', 'blackbox'):         {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow},
    ('mfcc', 'whitebox'):        {1: Conv_sp1_mfcc, 2: Conv_sp2_mfcc, 3: Conv_sp3_mfcc, 4: Conv_sp4_mfcc},
    ('mfcc', 'blackbox'):        {1: Conv_sp1_mfcc, 2: Conv_sp2_mfcc, 3: Conv_sp3_mfcc, 4: Conv_sp4_mfcc},
    ('mfcc', 'whitebox_shallow'): {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow, 4: Conv_sp4_shallow},
    ('mfcc', 'blackbox_shallow'): {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow, 4: Conv_sp4_shallow},
}

CKPT_DIR_MAP = {
    ('mel', 'whitebox'):          f'{BASE}/inversion/mel/whitebox',
    ('mel', 'whitebox_shallow'):  f'{BASE}/inversion/mel/whitebox_shallow',
    ('mel', 'blackbox'):          f'{BASE}/inversion/mel/blackbox',
    ('mfcc', 'whitebox'):         f'{BASE}/inversion/mfcc/whitebox',
    ('mfcc', 'blackbox'):         f'{BASE}/inversion/mfcc/blackbox',
    ('mfcc', 'whitebox_shallow'): f'{BASE}/inversion/mfcc/whitebox_shallow',
    ('mfcc', 'blackbox_shallow'): f'{BASE}/inversion/mfcc/blackbox_shallow',
}

EMO_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
EMO_INV = {v: k for k, v in EMO_MAP.items()}

MODEL_COMBOS = [
    ('mel', 'whitebox', [1,2,3,4]),
    ('mel', 'whitebox_shallow', [1,2,3,4]),
    ('mel', 'blackbox', [1,2,3]),
    ('mfcc', 'whitebox', [1,2,3,4]),
    ('mfcc', 'blackbox', [1,2,3,4]),
    ('mfcc', 'whitebox_shallow', [1,2,3,4]),
    ('mfcc', 'blackbox_shallow', [1,2,3,4]),
]

# ── Feature extraction ──────────────────────────────────────────────

def feat_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
                                          n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return _pad_or_trim(mel_db)

def feat_mfcc(y):
    m = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                              n_fft=N_FFT, hop_length=HOP_LENGTH)
    return _pad_or_trim(m)

def _pad_or_trim(feat):
    if feat.shape[1] > MAX_TIME:
        return feat[:, :MAX_TIME]
    import scipy.interpolate as sci
    h, w = feat.shape
    padded = np.zeros((h, MAX_TIME))
    padded[:, :w] = feat
    for i in range(h):
        x_new = np.linspace(0, w - 1, MAX_TIME)
        x_old = np.arange(w)
        fi = sci.interp1d(x_old, feat[i], kind='linear', fill_value='extrapolate')
        padded[i] = fi(x_new)
    return padded

def normalize(feat, mn, mx):
    norm = ((feat - mn) / (mx - mn)) * 255
    return torch.from_numpy(np.uint8(norm)).unsqueeze(0).float() / 255.0

def denormalize(tensor, mn, mx):
    arr = tensor.squeeze().cpu().numpy()
    return arr * (mx - mn) + mn

def compute_snr(orig, recon):
    noise_pow = torch.mean((orig - recon) ** 2)
    signal_pow = torch.mean(orig ** 2)
    if noise_pow < 1e-12:
        return float('inf')
    return 10 * torch.log10(signal_pow / noise_pow).item()

def reconstruct_audio_from_mel(mel_db_np, sr=SAMPLE_RATE, n_fft=N_FFT, hop=HOP_LENGTH):
    mel_power = librosa.db_to_power(mel_db_np)
    audio = librosa.feature.inverse.mel_to_audio(
        mel_power, sr=sr, n_fft=n_fft, hop_length=hop,
        power=2.0, n_iter=32
    )
    return audio

def reconstruct_audio_from_mfcc(mfcc_np, sr=SAMPLE_RATE, n_fft=N_FFT, hop=HOP_LENGTH, n_mels=N_MELS):
    mel_power = librosa.feature.inverse.mfcc_to_mel(mfcc_np, n_mels=n_mels)
    mel_power = np.nan_to_num(mel_power, nan=0.0, posinf=0.0, neginf=0.0)
    mel_power = np.maximum(mel_power, 1e-10)
    try:
        audio = librosa.feature.inverse.mel_to_audio(
            mel_power, sr=sr, n_fft=n_fft, hop_length=hop,
            power=2.0, n_iter=32
        )
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio
    except Exception:
        return None

# ── Load test set file list ─────────────────────────────────────────

test_files = sorted(glob(f'{PROCESSED}/test/*_mel.npy'))
test_samples = []
for fp in test_files:
    basename = os.path.basename(fp).replace('_mel.npy', '.wav')
    wav_path = os.path.join(WAV_DIR, basename)
    if not os.path.exists(wav_path):
        continue
    emo = basename.split('_')[2]
    label = EMO_MAP.get(emo, -1)
    if label < 0:
        continue
    test_samples.append({
        'fname': os.path.basename(fp).replace('_mel.npy', ''),
        'wav_path': wav_path,
        'label': label,
    })
print(f'Test samples: {len(test_samples)}', flush=True)

# ── Select WER subset ───────────────────────────────────────────────

random.seed(42)
wer_indices = set(random.sample(range(len(test_samples)), min(WER_SUBSET, len(test_samples))))
audio_indices = set(random.sample(list(wer_indices), min(AUDIO_SAMPLES, len(wer_indices))))

# ── Load classifier ─────────────────────────────────────────────────

def load_classifier(feat_type):
    cl_path = f'{BASE}/classifier/{feat_type}_train_record/classifier.pth'
    ckpt = torch.load(cl_path, map_location='cpu')
    model = ResNet18(BasicBlock, 1, 6).to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

# ── Transcribe clean audio (WER subset) ─────────────────────────────

print('Loading Whisper model...', flush=True)
asr = whisper.load_model('small', device='cuda' if torch.cuda.is_available() else 'cpu')
print('Whisper loaded.', flush=True)

clean_texts = {}
for idx in tqdm(wer_indices, desc='Transcribing clean audio'):
    samp = test_samples[idx]
    audio, _ = librosa.load(samp['wav_path'], sr=SAMPLE_RATE, mono=True)
    result = asr.transcribe(audio, language='en', task='transcribe', fp16=torch.cuda.is_available())
    clean_texts[idx] = result['text'].strip()

# ── Run evaluation ──────────────────────────────────────────────────

all_rows = []
wer_rows = []

for feat_type, mode, split_points in MODEL_COMBOS:
    print(f'\n{"="*70}\n{feat_type.upper()} / {mode}\n{"="*70}', flush=True)

    classifier = load_classifier(feat_type)
    mn, mx = (MEL_MIN, MEL_MAX) if feat_type == 'mel' else (MFCC_MIN, MFCC_MAX)
    feat_fn = feat_mel if feat_type == 'mel' else feat_mfcc
    inv_cls_map = INV_CLS_MAP[(feat_type, mode)]
    ckpt_dir = CKPT_DIR_MAP[(feat_type, mode)]

    os.makedirs(f'{OUTPUT_DIR}/spectrograms/{feat_type}_{mode}', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/audio_samples/{feat_type}_{mode}', exist_ok=True)

    for sp in split_points:
        print(f'\n--- Split Point {sp} ---', flush=True)

        inversion = inv_cls_map[sp]().to(DEVICE)
        ckpt_path = f'{ckpt_dir}/split_point{sp}/train_record/inversion.pth'
        if not os.path.exists(ckpt_path):
            print(f'  WARNING: checkpoint not found at {ckpt_path}, skipping', flush=True)
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu')
        inversion.load_state_dict(ckpt['model'])
        inversion.eval()

        mse_list, snr_list = [], []
        corr_inp_list, corr_rec_list = [], []

        for idx, samp in enumerate(tqdm(test_samples, desc=f'eval {feat_type}/{mode}/sp{sp}')):
            audio, _ = librosa.load(samp['wav_path'], sr=SAMPLE_RATE, mono=True)
            feat_db = feat_fn(audio)
            inp = normalize(feat_db, mn, mx).unsqueeze(0).to(DEVICE)
            label = samp['label']

            with torch.no_grad():
                logit_inp = classifier(inp, split_point=0)
                ft = classifier(inp, split_point=sp)
                rec = inversion(ft)
                logit_rec = classifier(rec, split_point=0)

            corr_inp = (logit_inp.argmax(dim=1).item() == label)
            corr_rec = (logit_rec.argmax(dim=1).item() == label)

            mse = F.mse_loss(rec, inp).item()
            snr = compute_snr(inp, rec)

            mse_list.append(mse)
            snr_list.append(snr)
            corr_inp_list.append(int(corr_inp))
            corr_rec_list.append(int(corr_rec))

            all_rows.append({
                'feature': feat_type, 'mode': mode, 'split_point': sp,
                'sample': samp['fname'], 'label': EMO_INV[label],
                'mse': mse, 'snr': snr,
                'correct_input': int(corr_inp),
                'correct_recon': int(corr_rec),
            })

            # ── WER ──
            if idx in wer_indices:
                try:
                    if feat_type == 'mel':
                        recon_db = denormalize(rec, mn, mx)
                        recon_audio = reconstruct_audio_from_mel(recon_db)
                    else:
                        recon_mfcc = denormalize(rec, mn, mx)
                        recon_audio = reconstruct_audio_from_mfcc(recon_mfcc)
                    result = asr.transcribe(recon_audio, language='en', task='transcribe',
                                            fp16=torch.cuda.is_available())
                    recon_text = result['text'].strip()
                    clean_text = clean_texts[idx]
                    wer_val = jiwer.wer(clean_text, recon_text)

                    wer_rows.append({
                        'feature': feat_type, 'mode': mode, 'split_point': sp,
                        'sample': samp['fname'], 'label': EMO_INV[label],
                        'clean_text': clean_text,
                        'recon_text': recon_text,
                        'wer': wer_val,
                    })

                    # ── Save audio samples ──
                    if idx in audio_indices:
                        sf.write(
                            f'{OUTPUT_DIR}/audio_samples/{feat_type}_{mode}/clean_sp{sp}_{samp["fname"]}.wav',
                            audio, SAMPLE_RATE
                        )
                        sf.write(
                            f'{OUTPUT_DIR}/audio_samples/{feat_type}_{mode}/recon_sp{sp}_{samp["fname"]}.wav',
                            recon_audio, SAMPLE_RATE
                        )
                except Exception as e:
                    print(f'  WER failed for {samp["fname"]} sp{sp}: {e}', flush=True)

            # ── Spectrogram visualization (every 50) ──
            if idx % 50 == 0:
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                for ax, img, ti in zip(axes,
                    [inp.cpu(), rec.cpu()],
                    ['Original', 'Recon']):
                    show = denormalize(img, mn, mx)
                    y_axis = 'mel' if feat_type == 'mel' else 'linear'
                    librosa.display.specshow(show, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                             x_axis='time', y_axis=y_axis, ax=ax, cmap='magma')
                    ax.set_title(f'{ti} ({feat_type}/{mode}/sp{sp})')
                plt.tight_layout()
                plt.savefig(f'{OUTPUT_DIR}/spectrograms/{feat_type}_{mode}/sp{sp}_{idx:04d}.png',
                            dpi=150, bbox_inches='tight')
                plt.close()

        n = len(test_samples)
        print(f'  MSE={np.mean(mse_list):.6f}  SNR={np.mean(snr_list):.2f}  '
              f'ACC_inp={np.mean(corr_inp_list)*100:.1f}%  ACC_rec={np.mean(corr_rec_list)*100:.1f}%',
              flush=True)

# ── Save results ────────────────────────────────────────────────────

df = pd.DataFrame(all_rows)
df.to_csv(f'{OUTPUT_DIR}/eval_detail.csv', index=False)
print(f'\nSaved {OUTPUT_DIR}/eval_detail.csv', flush=True)

if wer_rows:
    df_wer = pd.DataFrame(wer_rows)
    df_wer.to_csv(f'{OUTPUT_DIR}/wer_detail.csv', index=False)
    print(f'Saved {OUTPUT_DIR}/wer_detail.csv', flush=True)

# Summary
summary_rows = []
for (feat, mode, sp), grp in df.groupby(['feature', 'mode', 'split_point']):
    sub_wer = df_wer[(df_wer['feature'] == feat) & (df_wer['mode'] == mode) & (df_wer['split_point'] == sp)] if 'df_wer' in dir() else pd.DataFrame()
    summary_rows.append({
        'feature': feat, 'mode': mode, 'split_point': sp,
        'mse_mean': grp['mse'].mean(), 'mse_std': grp['mse'].std(),
        'snr_mean': grp['snr'].mean(), 'snr_std': grp['snr'].std(),
        'acc_input': grp['correct_input'].mean() * 100,
        'acc_recon': grp['correct_recon'].mean() * 100,
        'acc_drop': (grp['correct_input'].mean() - grp['correct_recon'].mean()) * 100,
        'wer_mean': sub_wer['wer'].mean() if len(sub_wer) > 0 else None,
    })
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(f'{OUTPUT_DIR}/eval_summary.csv', index=False)
print(f'Saved {OUTPUT_DIR}/eval_summary.csv', flush=True)

print('\n' + '=' * 70)
print(df_summary.to_string(index=False))
print('=' * 70)

print('\nDone!', flush=True)
