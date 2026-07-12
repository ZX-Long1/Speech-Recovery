from __future__ import print_function
import torch
from torchvision import transforms
import os
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import csv
from PIL import Image
import scipy.interpolate as sci
import time

from Model2 import Classifier, Conv_sp3

MAX_VAL = 9.5367431640625e-07
MIN_VAL = -80.00000381469727
SAMPLE_RATE = 25250
N_MELS = 64
MAX_LENGTH = 200
SPLIT_POINT = 3
NOISE_AMPLITUDE = 0.3
NUM_SAMPLES = 20
CUDA_ID = 0

DATASET_ROOT = '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K/audio'
OUTPUT_DIR = './noise_evaluation_results'

MODEL_EPOCHS = {
    0:   '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K/classifier/mel_spect_train_record/classifier.pth',
    51:  '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K/classifier/mel_spect_train_record/classifier_51.pth',
}
INV_PATHS = {
    'whitebox': '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K/inversion/split_point3/train_record/inversion.pth',
    'blackbox': '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K/blackbox_mel_spect_inversion/split_point3/train_record/inversion.pth',
}

def compute_snr(signal, error):
    signal = signal.flatten()
    error = error.flatten()
    snr = 10 * np.log10(np.sum(signal ** 2) / (np.sum(error ** 2) + 1e-8))
    return float(snr)

def load_noise(path, target_sr=SAMPLE_RATE):
    noise, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        noise = librosa.resample(noise, orig_sr=sr, target_sr=target_sr)
    noise = noise / (np.max(np.abs(noise)) + 1e-8)
    return noise

def add_noise(audio, noise, amp=NOISE_AMPLITUDE):
    if len(noise) < len(audio):
        noise = np.tile(noise, int(np.ceil(len(audio) / len(noise))))
    noise = noise[:len(audio)]
    audio_rms = np.sqrt(np.mean(audio ** 2))
    noise_rms = np.sqrt(np.mean(noise ** 2)) + 1e-8
    scaled_noise = noise * (audio_rms / noise_rms) * amp
    noisy = audio + scaled_noise
    noisy = noisy / (np.max(np.abs(noisy)) + 1e-8)
    return noisy

def extract_mel(audio, sr=SAMPLE_RATE, n_mels=N_MELS, max_len=MAX_LENGTH):
    framesize = int(0.025 * sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=framesize, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    if mel.shape[1] > max_len:
        return mel[:, :max_len]
    padded = np.zeros((mel.shape[0], max_len))
    padded[:, :mel.shape[1]] = mel
    for i in range(mel.shape[0]):
        x_new = np.linspace(0, mel.shape[1] - 1, max_len)
        x_old = np.arange(mel.shape[1])
        f = sci.interp1d(x_old, mel[i, :], kind='linear', fill_value='extrapolate')
        padded[i, :] = f(x_new)
    return padded

def normalize_mel(mel):
    norm = ((mel - MIN_VAL) / (MAX_VAL - MIN_VAL)) * 255
    return np.uint8(norm)

def denormalize_mel(x):
    return x.astype(np.float64) * (MAX_VAL - MIN_VAL) + MIN_VAL

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(f"cuda:{CUDA_ID}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(CUDA_ID)
    print(f"Device: {device}")
    print("Loading noise...")
    noise_audio = load_noise('./noise/20260601-0902.m4a', SAMPLE_RATE)
    print(f"  Noise: {len(noise_audio)} samples, {SAMPLE_RATE} Hz")
    transform = transforms.Compose([transforms.ToTensor()])
    print("Loading inversion models...")
    inv_models = {}
    for atk, path in INV_PATHS.items():
        ckpt = torch.load(path, map_location=device)
        inv = Conv_sp3().to(device)
        inv.load_state_dict(ckpt['model'])
        inv.eval()
        inv_models[atk] = inv
        print(f"  {atk}: best_acc={ckpt.get('best_acc', '?'):.8f}")
    test_wav_dir = os.path.join(DATASET_ROOT, 'test_set')
    test_mel_dir = os.path.join(DATASET_ROOT, 'test_set_mel_spect_numpy')
    wav_files = sorted(os.listdir(test_wav_dir))[:NUM_SAMPLES]
    all_data = []
    print(f"Preloading {NUM_SAMPLES} samples...")
    for wav_name in wav_files:
        wav_path = os.path.join(test_wav_dir, wav_name)
        npy_name = os.path.splitext(wav_name)[0] + '.npy'
        npy_path = os.path.join(test_mel_dir, npy_name)
        if not os.path.exists(npy_path):
            continue
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        noisy_audio = add_noise(audio, noise_audio)
        clean_mel = np.load(npy_path)
        noisy_mel = extract_mel(noisy_audio)
        clean_norm = normalize_mel(clean_mel)
        noisy_norm = normalize_mel(noisy_mel)
        t_clean = transform(Image.fromarray(clean_norm)).unsqueeze(0).to(device)
        t_noisy = transform(Image.fromarray(noisy_norm)).unsqueeze(0).to(device)
        all_data.append((wav_name, t_clean, t_noisy, clean_mel, noisy_mel))
    print(f"  Loaded {len(all_data)} samples\n")
    table_rows = []
    for epoch_name, ckpt_path in sorted(MODEL_EPOCHS.items()):
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] epoch {epoch_name}")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        cl = Classifier(nc=1, ndf=64, nz=10).to(device)
        cl.load_state_dict(ckpt['model'])
        cl.eval()
        acc = ckpt.get('best_cl_acc', ckpt.get('acc', '?'))
        print(f"Classifier epoch={epoch_name}, acc={acc}")
        for atk_name, inv in inv_models.items():
            results = []
            for wav_name, t_clean, t_noisy, clean_mel, noisy_mel in all_data:
                with torch.no_grad():
                    feat_clean = cl(t_clean, split_point=SPLIT_POINT)
                    feat_noisy = cl(t_noisy, split_point=SPLIT_POINT)
                    rec_clean = inv(feat_clean)
                    rec_noisy = inv(feat_noisy)
                rec_clean_np = rec_clean.cpu().squeeze().numpy()
                rec_noisy_np = rec_noisy.cpu().squeeze().numpy()
                ref_np = t_clean.cpu().squeeze().numpy()
                mse_clean = float(np.mean((rec_clean_np - ref_np) ** 2))
                mse_noisy = float(np.mean((rec_noisy_np - ref_np) ** 2))
                rec_clean_denorm = denormalize_mel(rec_clean_np)
                rec_noisy_denorm = denormalize_mel(rec_noisy_np)
                snr_clean = compute_snr(clean_mel, rec_clean_denorm - clean_mel)
                snr_noisy = compute_snr(clean_mel, rec_noisy_denorm - clean_mel)
                results.append([wav_name, mse_clean, mse_noisy, snr_clean, snr_noisy])
            avg_mse_c = np.mean([r[1] for r in results])
            avg_mse_n = np.mean([r[2] for r in results])
            avg_snr_c = np.mean([r[3] for r in results])
            avg_snr_n = np.mean([r[4] for r in results])
            print(f"  {atk_name:>10}: MSE(c)={avg_mse_c:.8f}  MSE(n)={avg_mse_n:.8f}  SNR(c)={avg_snr_c:.2f}dB  SNR(n)={avg_snr_n:.2f}dB")
            csv_path = os.path.join(OUTPUT_DIR, f'results_ep{epoch_name}_{atk_name}.csv')
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['sample', 'mse_clean', 'mse_noisy', 'snr_clean', 'snr_noisy'])
                w.writerows(results)
            table_rows.append((epoch_name, atk_name, f"{avg_mse_c:.8f}", f"{avg_mse_n:.8f}", f"{avg_snr_c:.2f}", f"{avg_snr_n:.2f}"))
    sl_path = os.path.join(OUTPUT_DIR, 'split_learning_summary.txt')
    with open(sl_path, 'w') as f:
        f.write(f"Split Learning - Noise Evaluation\n{'='*70}\n")
        f.write(f"Noise amplitude: {NOISE_AMPLITUDE}\nSplit point: {SPLIT_POINT}\nSamples: {NUM_SAMPLES}\n\n")
        f.write(f"{'Epoch':<8} {'Attack':<12} {'MSE(clean)':<16} {'MSE(noisy)':<16} {'SNR(clean)':<12} {'SNR(noisy)':<12}\n")
        f.write(f"{'-'*75}\n")
        for row in table_rows:
            f.write(f"{str(row[0]):<8} {row[1]:<12} {row[2]:<16} {row[3]:<16} {row[4]:<12} {row[5]:<12}\n")
    print(f"\nSplit learning summary -> {sl_path}")
    print("Done!")

if __name__ == '__main__':
    main()
