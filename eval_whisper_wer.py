import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from glob import glob
from tqdm import tqdm
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cremad_experiment.whisper_models import WhisperEncoder, Decoder_whitebox, Decoder_blackbox
from wer_core import gt_recon_wer_summary

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_point', type=int, required=True, choices=[1, 2, 3, 4, 8])
parser.add_argument('--mode', type=str, required=True, choices=['whitebox', 'blackbox'])
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--subset', type=int, default=0, help='Evaluate on subset (0=all)')
parser.add_argument('--base_dir', type=str, default='whisper_inversion',
                    help='Subdir of CREMA-D holding sp{sp}_{mode} outputs')
args = parser.parse_args()

DEVICE = torch.device(f'cuda:{args.device_id}')
sp = args.split_point
mode = args.mode
n_subset = args.subset
print(f'Evaluating: sp={sp} mode={mode} device={DEVICE}', flush=True)

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
WAV_DIR = f'{BASE}/AudioWAV'
OUT_DIR = f'{BASE}/{args.base_dir}/sp{sp}_{mode}'
os.makedirs(OUT_DIR, exist_ok=True)

test_files = sorted(glob(f'{BASE}/processed/test/*_mel.npy'))
if n_subset > 0:
    test_files = test_files[:n_subset]
print(f'Test samples: {len(test_files)}', flush=True)

print('Loading models...', flush=True)
import whisper
whisper_model = whisper.load_model('small').to(DEVICE)
whisper_model.eval()

enc = WhisperEncoder(split_point=sp, freeze=True).to(DEVICE)
enc.eval()

if mode == 'whitebox':
    decoder = Decoder_whitebox().to(DEVICE)
else:
    decoder = Decoder_blackbox().to(DEVICE)
decoder_path = f'{OUT_DIR}/decoder_best.pth'
decoder.load_state_dict(torch.load(decoder_path, map_location=DEVICE))
decoder.eval()
print(f'Loaded decoder from {decoder_path}', flush=True)

results_list = []
pbar = tqdm(test_files, desc=f'Evaluating sp{sp} {mode}')
for fp in pbar:
    stem = os.path.basename(fp).replace('_mel.npy', '')
    wav_path = os.path.join(WAV_DIR, f'{stem}.wav')
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)

    # Original mel
    mel = whisper.log_mel_spectrogram(audio, padding=0).unsqueeze(0).to(DEVICE)

    # Reconstruct
    with torch.no_grad():
        feat = enc(mel)
        recon = decoder(feat)
    t = min(recon.shape[-1], mel.shape[-1])
    recon_mel = recon[:, :, :t]

    # Convert recon mel (log10-scale) to audio for transcript
    recon_np = recon_mel.squeeze(0).cpu().numpy()
    power_mel = np.power(10.0, recon_np)
    S = librosa.feature.inverse.mel_to_stft(power_mel, sr=16000, n_fft=400, power=2.0)
    recon_audio = librosa.griffinlim(S, hop_length=160, n_iter=32)

    with torch.no_grad():
        hyp_result = whisper_model.transcribe(recon_audio, language='en', task='transcribe', fp16=True)
    hyp = hyp_result['text'].strip().lower()

    mse = F.mse_loss(recon_mel, mel[:, :, :t]).item()
    signal_power = mel[:, :, :t].pow(2).mean().item()
    snr_val = 10 * np.log10(signal_power / (mse + 1e-10))

    results_list.append({
        'file': stem, 'reference': '', 'hypothesis': hyp,
        'wer': 0.0, 'mse': mse, 'snr': snr_val,
    })
    pbar.set_postfix(MSE=f'{np.mean([r["mse"] for r in results_list]):.6f}')

ws = gt_recon_wer_summary(results_list, stem_key='file', hyp_key='hypothesis')
avg_mse = float(np.mean([r['mse'] for r in results_list]))
avg_snr = float(np.mean([r['snr'] for r in results_list]))
print(f'\n=== Results sp{sp} {mode} ===', flush=True)
print(f'GT->Recon corpus WER: {ws["corpus_wer_pct"]}%  95%CI=[{ws["ci_low"]}, {ws["ci_high"]}]  '
      f'(per-sample {ws["per_sample_wer_pct"]}%, n={ws["n"]})', flush=True)
print(f'Average MSE: {avg_mse:.6f}', flush=True)
print(f'Average SNR: {avg_snr:.2f} dB', flush=True)

df = pd.DataFrame(results_list)
csv_path = f'{OUT_DIR}/eval_results.csv'
df.to_csv(csv_path, index=False)
print(f'Saved: {csv_path}', flush=True)

summary = {
    'experiment': f'sp{sp}_{mode}',
    'split_point': sp, 'mode': mode,
    'samples': len(df),
    'corpus_wer_pct': ws['corpus_wer_pct'],
    'per_sample_wer_pct': ws['per_sample_wer_pct'],
    'ci_low': ws['ci_low'],
    'ci_high': ws['ci_high'],
    'avg_mse': round(avg_mse, 6),
    'avg_snr_db': round(avg_snr, 2),
}
summary_df = pd.DataFrame([summary])
summary_csv = f'{OUT_DIR}/eval_summary.csv'
summary_df.to_csv(summary_csv, index=False)
print(f'Saved: {summary_csv}', flush=True)
print(summary_df.to_string(index=False))
print('\nDone.', flush=True)
