import os, sys, json, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import jiwer
import whisper
from glob import glob
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cremad_experiment.cremad_models import BasicBlock, ResNet18, Conv_sp4_shallow

SAMPLE_RATE = 16000
N_MELS = 64
MAX_TIME = 640
N_FFT = 320
HOP_LENGTH = 80
BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
PROCESSED = f'{BASE}/processed'
WAV_DIR = f'{BASE}/AudioWAV'
OUTPUT_DIR = f'{BASE}/eval_results_mel_bb_sp4'
WER_SUBSET = 200
AUDIO_SAMPLES = 20
SPLIT_POINT = 4
MODE = 'blackbox'
FEAT_TYPE = 'mel'
EMO_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
EMO_INV = {v: k for k, v in EMO_MAP.items()}

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/spectrograms', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/audio_samples', exist_ok=True)
print(f'Device: {DEVICE}', flush=True)

norm_info = json.load(open(f'{PROCESSED}/norm_info.json'))
mn, mx = norm_info['mel_min'], norm_info['mel_max']

def feat_fn(y):
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

def normalize(feat, mn, mx):
    norm = ((feat - mn) / (mx - mn)) * 255
    return torch.from_numpy(np.uint8(norm)).unsqueeze(0).float() / 255.0

def denormalize(tensor, mn, mx):
    return tensor.squeeze().cpu().numpy() * (mx - mn) + mn

def compute_snr(orig, recon):
    noise = torch.mean((orig - recon) ** 2)
    signal = torch.mean(orig ** 2)
    if noise < 1e-12:
        return float('inf')
    return 10 * torch.log10(signal / noise).item()

# Load test samples
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
        'fname': basename.replace('.wav', ''),
        'wav_path': wav_path,
        'label': label,
    })
print(f'Test samples: {len(test_samples)}', flush=True)

random.seed(42)
wer_indices = set(random.sample(range(len(test_samples)), min(WER_SUBSET, len(test_samples))))
audio_indices = set(random.sample(list(wer_indices), min(AUDIO_SAMPLES, len(wer_indices))))

# Load classifier
classifier = ResNet18(BasicBlock, 1, 6).to(DEVICE)
cl_path = f'{BASE}/classifier/mel_train_record/classifier.pth'
ckpt = torch.load(cl_path, map_location='cpu')
classifier.load_state_dict(ckpt['model'])
classifier.eval()

# Load inversion model
inversion = Conv_sp4_shallow().to(DEVICE)
inv_path = f'{BASE}/inversion/mel/blackbox/split_point{SPLIT_POINT}/train_record/inversion.pth'
ckpt = torch.load(inv_path, map_location='cpu')
inversion.load_state_dict(ckpt['model'])
inversion.eval()
print(f'Loaded inversion (best epoch={ckpt.get("best_epoch","?")})', flush=True)

# Clean text transcription
print('Loading Whisper...', flush=True)
asr = whisper.load_model('small', device='cuda' if torch.cuda.is_available() else 'cpu')
clean_texts = {}
for idx in tqdm(wer_indices, desc='Clean transcriptions'):
    samp = test_samples[idx]
    audio, _ = librosa.load(samp['wav_path'], sr=SAMPLE_RATE, mono=True)
    result = asr.transcribe(audio, language='en', task='transcribe', fp16=torch.cuda.is_available())
    clean_texts[idx] = result['text'].strip()

# Evaluate
all_rows, wer_rows = [], []
mse_list, snr_list = [], []
corr_inp_list, corr_rec_list = [], []

for idx, samp in enumerate(tqdm(test_samples, desc=f'eval mel blackbox sp{SPLIT_POINT}')):
    audio, _ = librosa.load(samp['wav_path'], sr=SAMPLE_RATE, mono=True)
    feat_db = feat_fn(audio)
    inp = normalize(feat_db, mn, mx).unsqueeze(0).to(DEVICE)
    label = samp['label']

    with torch.no_grad():
        logit_inp = classifier(inp, split_point=0)
        ft = classifier(inp, split_point=SPLIT_POINT)
        rec = inversion(ft)
        logit_rec = classifier(rec, split_point=0)

    corr_inp = int(logit_inp.argmax(dim=1).item() == label)
    corr_rec = int(logit_rec.argmax(dim=1).item() == label)
    mse = F.mse_loss(rec, inp).item()
    snr = compute_snr(inp, rec)

    mse_list.append(mse)
    snr_list.append(snr)
    corr_inp_list.append(corr_inp)
    corr_rec_list.append(corr_rec)

    all_rows.append({
        'feature': FEAT_TYPE, 'mode': MODE, 'split_point': SPLIT_POINT,
        'sample': samp['fname'], 'label': EMO_INV[label],
        'mse': mse, 'snr': snr,
        'correct_input': corr_inp, 'correct_recon': corr_rec,
    })

    # WER
    if idx in wer_indices:
        try:
            recon_db = denormalize(rec, mn, mx)
            mel_power = librosa.db_to_power(recon_db)
            mel_power = np.clip(mel_power, 1e-10, None)
            recon_audio = librosa.feature.inverse.mel_to_audio(
                mel_power, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                power=2.0, n_iter=32
            )
            recon_audio = np.nan_to_num(recon_audio, nan=0.0)
            peak = np.max(np.abs(recon_audio))
            if peak > 0:
                recon_audio = recon_audio / peak
            result = asr.transcribe(recon_audio, language='en', task='transcribe', fp16=True)
            recon_text = result['text'].strip()
            clean_text = clean_texts[idx]
            wer_val = jiwer.wer(clean_text, recon_text)
            wer_rows.append({
                'feature': FEAT_TYPE, 'mode': MODE, 'split_point': SPLIT_POINT,
                'sample': samp['fname'], 'label': EMO_INV[label],
                'clean_text': clean_text, 'recon_text': recon_text, 'wer': wer_val,
            })
            if idx in audio_indices:
                sf.write(f'{OUTPUT_DIR}/audio_samples/clean_{samp["fname"]}.wav', audio, SAMPLE_RATE)
                sf.write(f'{OUTPUT_DIR}/audio_samples/recon_{samp["fname"]}.wav', recon_audio, SAMPLE_RATE)
        except Exception as e:
            print(f'  WER failed for {samp["fname"]}: {e}', flush=True)

    # Spectrogram visualization
    if idx % 50 == 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, img, ti in zip(axes, [inp.cpu(), rec.cpu()], ['Original', 'Recon']):
            show = denormalize(img, mn, mx)
            librosa.display.specshow(show, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                     x_axis='time', y_axis='mel', ax=ax, cmap='magma')
            ax.set_title(f'{ti} (mel/blackbox/sp{SPLIT_POINT})')
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/spectrograms/sp{SPLIT_POINT}_{idx:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()

# Results
n = len(test_samples)
print(f'\nMSE={np.mean(mse_list):.6f}  SNR={np.mean(snr_list):.2f}  '
      f'ACC_inp={np.mean(corr_inp_list)*100:.1f}%  ACC_rec={np.mean(corr_rec_list)*100:.1f}%', flush=True)

import pandas as pd
pd.DataFrame(all_rows).to_csv(f'{OUTPUT_DIR}/eval_detail.csv', index=False)
if wer_rows:
    pd.DataFrame(wer_rows).to_csv(f'{OUTPUT_DIR}/wer_detail.csv', index=False)

# Summary text
with open(f'{OUTPUT_DIR}/eval_summary.txt', 'w') as f:
    f.write(f'mel/blackbox/sp{SPLIT_POINT} Evaluation\n')
    f.write(f'Samples: {n}\n')
    f.write(f'MSE: {np.mean(mse_list):.6f} ± {np.std(mse_list):.6f}\n')
    f.write(f'SNR: {np.mean(snr_list):.2f} ± {np.std(snr_list):.2f} dB\n')
    f.write(f'Accuracy (input):  {np.mean(corr_inp_list)*100:.1f}%\n')
    f.write(f'Accuracy (recon):  {np.mean(corr_rec_list)*100:.1f}%\n')
    if wer_rows:
        wers = [r['wer'] for r in wer_rows]
        f.write(f'WER: {np.mean(wers):.4f} ± {np.std(wers):.4f} (n={len(wers)})\n')
    else:
        f.write(f'WER: N/A\n')

print(f'Saved to {OUTPUT_DIR}', flush=True)
print('Done!', flush=True)
