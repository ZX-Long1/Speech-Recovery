import os, sys, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
from glob import glob
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cremad_experiment.whisper_models import WhisperEncoder, Decoder_whitebox, Decoder_blackbox, tvd_loss

parser = argparse.ArgumentParser()
parser.add_argument('--split_point', type=int, required=True, choices=[1, 2, 4, 8])
parser.add_argument('--mode', type=str, required=True, choices=['whitebox', 'blackbox'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--tvd_lam', type=float, default=0.5)
parser.add_argument('--device_id', type=int, default=0)
args = parser.parse_args()

DEVICE = torch.device(f'cuda:{args.device_id}')
sp = args.split_point
mode = args.mode
print(f'Config: split_point={sp} mode={mode} device={DEVICE}')

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
WAV_DIR = f'{BASE}/AudioWAV'
OUT_DIR = f'{BASE}/whisper_inversion/sp{sp}_{mode}'
os.makedirs(OUT_DIR, exist_ok=True)

class WhisperCremaDataset(Dataset):
    def __init__(self, npy_dir):
        self.files = sorted(glob(f'{npy_dir}/*_mel.npy'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        stem = os.path.basename(self.files[idx]).replace('_mel.npy', '')
        wav_path = os.path.join(WAV_DIR, f'{stem}.wav')
        audio, _ = librosa.load(wav_path, sr=16000, mono=True)
        audio = torch.from_numpy(audio).float()
        return audio

def whisper_log_mel(audio, sr=16000):
    audio_np = audio.numpy()
    mel = whisper_log_mel_spectrogram(audio_np)
    return mel

import whisper
def whisper_log_mel_spectrogram(audio):
    mel = whisper.log_mel_spectrogram(audio, padding=0)
    return mel  # (80, T)

def collate_fn(batch):
    max_len = max(a.shape[0] for a in batch)
    padded = []
    for a in batch:
        n = max_len - a.shape[0]
        padded.append(F.pad(a, (0, n)))
    return torch.stack(padded)

train_dataset = WhisperCremaDataset(f'{BASE}/processed/aux')
test_dataset = WhisperCremaDataset(f'{BASE}/processed/test')
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
print(f'Train: {len(train_dataset)}  Test: {len(test_dataset)}')

print('Loading Whisper encoder...')
encoder = WhisperEncoder(split_point=sp, freeze=True).to(DEVICE)
encoder.eval()

if mode == 'whitebox':
    decoder = Decoder_whitebox().to(DEVICE)
else:
    decoder = Decoder_blackbox().to(DEVICE)

optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

best_mse = float('inf')
for epoch in range(1, args.epochs + 1):
    decoder.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}', leave=False):
        batch = batch.to(DEVICE)
        mels = []
        for i in range(batch.shape[0]):
            audio_np = batch[i].cpu().numpy()
            mel = whisper_log_mel_spectrogram(audio_np)
            mels.append(mel)
        max_t = max(m.shape[1] for m in mels)
        mel_batch = torch.zeros(len(mels), 80, max_t, device=DEVICE)
        for i, m in enumerate(mels):
            mel_batch[i, :, :m.shape[1]] = m.to(DEVICE)

        with torch.no_grad():
            feat = encoder(mel_batch)

        recon = decoder(feat)
        t = min(recon.shape[-1], mel_batch.shape[-1])
        recon = recon[:, :, :t]
        target = mel_batch[:, :, :t]
        loss_mse = F.mse_loss(recon, target)
        loss_tvd = tvd_loss(recon, target)
        loss = loss_mse + args.tvd_lam * loss_tvd

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    scheduler.step()

    decoder.eval()
    test_mse = 0
    test_snr = 0
    n_test = 0
    with torch.no_grad():
        for batch in test_loader:
            audio_np = batch[0].cpu().numpy()
            mel = whisper_log_mel_spectrogram(audio_np).unsqueeze(0).to(DEVICE)
            feat = encoder(mel)
            recon = decoder(feat)
            t = min(recon.shape[-1], mel.shape[-1])
            recon = recon[:, :, :t]
            mel_t = mel[:, :, :t]
            mse = F.mse_loss(recon, mel_t).item()
            signal_power = mel_t.pow(2).mean().item()
            noise_power = mse
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
            test_mse += mse
            test_snr += snr
            n_test += 1

    test_mse /= n_test
    test_snr /= n_test
    print(f'Epoch {epoch}: train_loss={train_loss/len(train_loader):.6f}  test_mse={test_mse:.6f}  test_snr={test_snr:.2f}dB')

    if test_mse < best_mse:
        best_mse = test_mse
        torch.save(decoder.state_dict(), f'{OUT_DIR}/decoder_best.pth')
        print(f'  -> saved best decoder (mse={test_mse:.6f})')

torch.save(decoder.state_dict(), f'{OUT_DIR}/decoder_final.pth')
print(f'Done. Best MSE: {best_mse:.6f}')
