"""Diagnose: compute WER from original MFCC → audio (no inversion)."""
import os, sys, json, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import librosa
import soundfile as sf
import jiwer
import whisper
from glob import glob
from tqdm import tqdm

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
PROCESSED = f'{BASE}/processed'
WAV_DIR = f'{BASE}/AudioWAV'
SAMPLE_RATE = 16000
N_FFT = 320
HOP_LENGTH = 80
N_MELS = 64
MAX_TIME = 640
NUM_SAMPLES = 200
OUTPUT = f'{BASE}/eval_results/diagnose_mfcc_wer.csv'

norm_info = json.load(open(f'{PROCESSED}/norm_info.json'))
mn, mx = norm_info['mfcc_min'], norm_info['mfcc_max']

test_files = sorted(glob(f'{PROCESSED}/test/*_mfcc.npy'))
random.seed(42)
selected = random.sample(test_files, min(NUM_SAMPLES, len(test_files)))

asr = whisper.load_model('small', device='cuda')
print(f'Testing {len(selected)} samples...', flush=True)

rows = []
for fp in tqdm(selected):
    basename = os.path.basename(fp).replace('_mfcc.npy', '')
    wav_path = os.path.join(WAV_DIR, basename + '.wav')
    if not os.path.exists(wav_path):
        continue

    # Clean audio transcription
    audio_clean, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    result = asr.transcribe(audio_clean, language='en', task='transcribe', fp16=True)
    clean_text = result['text'].strip()

    # Load original MFCC (no inversion)
    mfcc_np = np.load(fp)

    # Denormalize
    feat = mfcc_np * (mx - mn) + mn

    # MFCC → Mel power
    mel_power = librosa.feature.inverse.mfcc_to_mel(feat, n_mels=N_MELS)
    mel_power = np.nan_to_num(mel_power, nan=0.0, posinf=0.0, neginf=0.0)
    mel_power = np.maximum(mel_power, 1e-10)
    try:
        audio_recon = librosa.feature.inverse.mel_to_audio(
            mel_power, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
            power=2.0, n_iter=32
        )
        audio_recon = np.nan_to_num(audio_recon, nan=0.0)
        peak = np.max(np.abs(audio_recon))
        if peak > 0:
            audio_recon = audio_recon / peak
    except Exception:
        # If Griffin-Lim fails entirely, skip this sample
        continue

    # Transcribe reconstructed audio
    result = asr.transcribe(audio_recon, language='en', task='transcribe', fp16=True)
    recon_text = result['text'].strip()

    wer = jiwer.wer(clean_text, recon_text)
    rows.append({
        'sample': basename,
        'clean_text': clean_text,
        'recon_text': recon_text,
        'wer': wer,
    })

avg_wer = np.mean([r['wer'] for r in rows])
print(f'\nAverage WER (raw MFCC→audio): {avg_wer:.4f} ({avg_wer*100:.1f}%)', flush=True)

import pandas as pd
df = pd.DataFrame(rows)
df.to_csv(OUTPUT, index=False)
print(f'Saved {OUTPUT}', flush=True)
