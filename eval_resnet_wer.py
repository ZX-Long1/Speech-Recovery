import os, sys, json, warnings
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cremad_experiment.cremad_models import ResNet18, BasicBlock
from cremad_experiment.cremad_models import Conv_sp1_shallow, Conv_sp3_shallow
from wer_core import gt_recon_wer_summary

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--split_point', type=int, required=True, choices=[1, 3])
parser.add_argument('--ckpt', type=str, required=True,
                    help='Path to inversion.pth (shallow decoder)')
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--subset', type=int, default=0, help='Evaluate on subset (0=all)')
parser.add_argument('--out_dir', type=str, required=True,
                    help='Dir to save eval_results.csv')
parser.add_argument('--mode', type=str, default='whitebox', choices=['whitebox', 'blackbox'],
                    help='Label only, for CSV naming')
args = parser.parse_args()

DEVICE = torch.device(f'cuda:{args.device_id}')
sp = args.split_point
print(f'Evaluating ResNet18 sp={sp} ckpt={args.ckpt}', flush=True)

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
WAV_DIR = f'{BASE}/AudioWAV'
SAMPLE_RATE = 16000
N_FFT = 320
HOP_LENGTH = 80
N_MELS = 64
MAX_TIME = 640
os.makedirs(args.out_dir, exist_ok=True)

norm_info = json.load(open(f'{BASE}/processed/norm_info.json'))
MEL_MIN, MEL_MAX = norm_info['mel_min'], norm_info['mel_max']

test_files = sorted(glob(f'{BASE}/processed/test/*_mel.npy'))
if args.subset > 0:
    test_files = test_files[:args.subset]
print(f'Test samples: {len(test_files)}', flush=True)

print('Loading models...', flush=True)
import whisper
whisper_model = whisper.load_model('small').to(DEVICE)
whisper_model.eval()

classifier = ResNet18(BasicBlock, 1, 6).to(DEVICE)
cl_path = f'{BASE}/classifier/mel_train_record/classifier.pth'
ck = torch.load(cl_path, map_location='cpu')
classifier.load_state_dict(ck['model'])
classifier.eval()

decoder = {1: Conv_sp1_shallow, 3: Conv_sp3_shallow}[sp]().to(DEVICE)
ck = torch.load(args.ckpt, map_location='cpu')
decoder.load_state_dict(ck['model'])
decoder.eval()


def feat_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
                                          n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] > MAX_TIME:
        return mel_db[:, :MAX_TIME]
    import scipy.interpolate as sci
    h, w = mel_db.shape
    padded = np.zeros((h, MAX_TIME))
    padded[:, :w] = mel_db
    for i in range(h):
        fi = sci.interp1d(np.arange(w), mel_db[i], kind='linear',
                          fill_value='extrapolate')
        padded[i] = fi(np.linspace(0, w - 1, MAX_TIME))
    return padded


def normalize(feat, mn, mx):
    return torch.from_numpy(np.uint8(((feat - mn) / (mx - mn)) * 255)) \
        .unsqueeze(0).float() / 255.0


def denormalize(tensor, mn, mx):
    return tensor.squeeze().cpu().numpy() * (mx - mn) + mn


def recon_mel_to_audio(mel_db_np):
    power = librosa.db_to_power(mel_db_np)
    return librosa.feature.inverse.mel_to_audio(power, sr=SAMPLE_RATE, n_fft=N_FFT,
                                                hop_length=HOP_LENGTH, power=2.0,
                                                n_iter=32)


results_list = []
pbar = tqdm(test_files, desc=f'Evaluating sp{sp}')
for fp in pbar:
    stem = os.path.basename(fp).replace('_mel.npy', '')
    wav_path = os.path.join(WAV_DIR, f'{stem}.wav')
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)

    feat_db = feat_mel(audio)
    inp = normalize(feat_db, MEL_MIN, MEL_MAX).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        ft = classifier(inp, split_point=sp)
        rec = decoder(ft)
    recon_db = denormalize(rec, MEL_MIN, MEL_MAX)
    recon_audio = recon_mel_to_audio(recon_db)

    with torch.no_grad():
        hyp_result = whisper_model.transcribe(recon_audio, language='en',
                                              task='transcribe', fp16=True)
    hyp = hyp_result['text'].strip().lower()

    mse = F.mse_loss(rec, inp).item()
    signal_power = inp.pow(2).mean().item()
    snr_val = 10 * np.log10(signal_power / (mse + 1e-10))

    results_list.append({'file': stem, 'reference': '', 'hypothesis': hyp,
                         'wer': 0.0, 'mse': mse, 'snr': snr_val})
    pbar.set_postfix(MSE=f'{np.mean([r["mse"] for r in results_list]):.6f}')

ws = gt_recon_wer_summary(results_list, stem_key='file', hyp_key='hypothesis')
avg_mse = float(np.mean([r['mse'] for r in results_list]))
avg_snr = float(np.mean([r['snr'] for r in results_list]))
print(f'\n=== Results sp{sp} ===', flush=True)
print(f'GT->Recon corpus WER: {ws["corpus_wer_pct"]}%  95%CI=[{ws["ci_low"]}, {ws["ci_high"]}]  '
      f'(per-sample {ws["per_sample_wer_pct"]}%, n={ws["n"]})', flush=True)
print(f'Average MSE: {avg_mse:.6f}', flush=True)
print(f'Average SNR: {avg_snr:.2f} dB', flush=True)

df = pd.DataFrame(results_list)
csv_path = f'{args.out_dir}/eval_results.csv'
df.to_csv(csv_path, index=False)
print(f'Saved: {csv_path}', flush=True)

summary = {
    'experiment': f'sp{sp}',
    'split_point': sp,
    'samples': len(df),
    'corpus_wer_pct': ws['corpus_wer_pct'],
    'per_sample_wer_pct': ws['per_sample_wer_pct'],
    'ci_low': ws['ci_low'],
    'ci_high': ws['ci_high'],
    'avg_mse': round(avg_mse, 6),
    'avg_snr_db': round(avg_snr, 2),
}
summary_df = pd.DataFrame([summary])
summary_csv = f'{args.out_dir}/eval_summary.csv'
summary_df.to_csv(summary_csv, index=False)
print(f'Saved: {summary_csv}', flush=True)
print(summary_df.to_string(index=False))
print('\nDone.', flush=True)
