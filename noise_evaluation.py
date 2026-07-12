from __future__ import print_function
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from PIL import Image
import scipy.interpolate as sci

from Model2 import Classifier, Conv_sp4, Conv_sp3, Conv_sp2, Conv_sp1

# ========== Constants ==========
MAX_VAL = 9.5367431640625e-07
MIN_VAL = -80.00000381469727
SAMPLE_RATE = 25250
N_MELS = 64
MAX_LENGTH = 200
NOISE_PATH = './noise/20260601-0902.m4a'
SPLIT_POINT = 3
NOISE_AMPLITUDE = 0.3
NUM_SAMPLES = 50
CUDA_ID = 0

DATASET_ROOT = '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K/audio'
BEST_CLASSIFIER_PATH = '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K/classifier/mel_spect_train_record/classifier.pth'
WHITEBOX_INV_PATH = '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K/inversion/split_point3/train_record/inversion.pth'
BLACKBOX_INV_PATH = '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K/blackbox_mel_spect_inversion/split_point3/train_record/inversion.pth'

CLASSIFIER_EPOCHS = {
    0:   '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K/classifier/mel_spect_train_record/classifier_51.pth',
    51:  '/media/sda1/zxlong/L_minghao/InverCRS/UrbanSound8K/UrbanSound8K/classifier/mel_spect_train_record/classifier_51.pth',
}

OUTPUT_DIR = './noise_evaluation_results'

# ========== Helper Functions ==========

def compute_snr(signal, error):
    signal = signal.flatten()
    error = error.flatten()
    num = np.sum(signal ** 2)
    den = np.sum(error ** 2) + 1e-8
    return 10 * np.log10(num / den)

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
    return noisy, scaled_noise

def extract_mel(audio, sr=SAMPLE_RATE, n_mels=N_MELS, max_len=MAX_LENGTH):
    framesize = int(0.025 * sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=framesize, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    if mel.shape[1] > max_len:
        mel = mel[:, :max_len]
    else:
        padded = np.zeros((mel.shape[0], max_len))
        padded[:, :mel.shape[1]] = mel
        for i in range(mel.shape[0]):
            x_new = np.linspace(0, mel.shape[1] - 1, max_len)
            x_old = np.arange(mel.shape[1])
            f = sci.interp1d(x_old, mel[i, :], kind='linear', fill_value='extrapolate')
            padded[i, :] = f(x_new)
        mel = padded
    return mel

def normalize_mel(mel):
    norm = ((mel - MIN_VAL) / (MAX_VAL - MIN_VAL)) * 255
    return np.uint8(norm)

def denormalize_mel(x):
    return x.astype(np.float64) * (MAX_VAL - MIN_VAL) + MIN_VAL

def get_inversion_model(sp, device):
    cls_map = {1: Conv_sp1, 2: Conv_sp2, 3: Conv_sp3, 4: Conv_sp4}
    return cls_map[sp]().to(device)

def load_classifier(path, device):
    ckpt = torch.load(path, map_location=device)
    model = Classifier(nc=1, ndf=64, nz=10).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    epoch = ckpt.get('epoch', '?')
    acc = ckpt.get('best_cl_acc', ckpt.get('acc', '?'))
    return model, epoch, acc

def load_inversion(path, sp, device):
    ckpt = torch.load(path, map_location=device)
    model = get_inversion_model(sp, device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt.get('best_acc', '?')

# ========== Plotting ==========

def save_comparison(clean_mel, noisy_mel, clean_rec, noisy_rec, save_path, name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = ['Original Clean', 'Noisy Input', 'Clean Reconstruction', 'Noisy Reconstruction']
    imgs = [clean_mel, noisy_mel, clean_rec, noisy_rec]
    for ax, t, img in zip(axes.flatten(), titles, imgs):
        librosa.display.specshow(img, sr=SAMPLE_RATE, x_axis='time', y_axis='mel', hop_length=512, cmap='magma', ax=ax)
        ax.set_title(t)
        ax.set_ylabel('Mel Frequency')
        ax.set_xlabel('Time (s)')
    plt.suptitle(f'Sample: {name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_mfcc_comparison(clean_m, noisy_m, clean_r, noisy_r, save_path, name):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = ['Original Clean MFCC', 'Noisy Input MFCC', 'Clean Recon MFCC', 'Noisy Recon MFCC']
    imgs = [clean_m, noisy_m, clean_r, noisy_r]
    for ax, t, img in zip(axes.flatten(), titles, imgs):
        librosa.display.specshow(img, sr=SAMPLE_RATE, x_axis='time', hop_length=512, cmap='magma', ax=ax)
        ax.set_title(t)
        ax.set_ylabel('MFCC')
        ax.set_xlabel('Time (s)')
    plt.suptitle(f'Sample: {name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ========== Evaluation ==========

def evaluate(classifier, inversion, noise_audio, device, tag='', out_dir=OUTPUT_DIR):
    classifier.eval()
    inversion.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    test_wav_dir = os.path.join(DATASET_ROOT, 'test_set')
    test_mel_dir = os.path.join(DATASET_ROOT, 'test_set_mel_spect_numpy')
    wav_files = sorted(os.listdir(test_wav_dir))[:NUM_SAMPLES]
    results = []
    spec_dir = os.path.join(out_dir, 'spectrograms', tag) if tag else os.path.join(out_dir, 'spectrograms')
    mfcc_dir = os.path.join(out_dir, 'mfcc', tag) if tag else os.path.join(out_dir, 'mfcc')
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(mfcc_dir, exist_ok=True)
    for wav_name in wav_files:
        wav_path = os.path.join(test_wav_dir, wav_name)
        npy_name = os.path.splitext(wav_name)[0] + '.npy'
        npy_path = os.path.join(test_mel_dir, npy_name)
        if not os.path.exists(npy_path):
            continue
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        noisy_audio, _ = add_noise(audio, noise_audio)
        clean_mel = np.load(npy_path)
        noisy_mel = extract_mel(noisy_audio)
        clean_norm = normalize_mel(clean_mel)
        noisy_norm = normalize_mel(noisy_mel)
        t_clean = transform(Image.fromarray(clean_norm)).unsqueeze(0).to(device)
        t_noisy = transform(Image.fromarray(noisy_norm)).unsqueeze(0).to(device)
        with torch.no_grad():
            feat_clean = classifier(t_clean, split_point=SPLIT_POINT)
            feat_noisy = classifier(t_noisy, split_point=SPLIT_POINT)
            rec_clean = inversion(feat_clean)
            rec_noisy = inversion(feat_noisy)
        rec_clean_np = rec_clean.cpu().squeeze().numpy()
        rec_noisy_np = rec_noisy.cpu().squeeze().numpy()
        clean_ref_np = t_clean.cpu().squeeze().numpy()
        mse_clean = np.mean((rec_clean_np - clean_ref_np) ** 2)
        mse_noisy = np.mean((rec_noisy_np - clean_ref_np) ** 2)
        mse_noisy_in = np.mean((rec_noisy_np - t_noisy.cpu().squeeze().numpy()) ** 2)
        clean_denorm = clean_mel
        rec_clean_denorm = denormalize_mel(rec_clean_np)
        rec_noisy_denorm = denormalize_mel(rec_noisy_np)
        snr_clean = compute_snr(clean_denorm, rec_clean_denorm - clean_denorm)
        snr_noisy = compute_snr(clean_denorm, rec_noisy_denorm - clean_denorm)
        results.append({
            'sample': wav_name,
            'mse_clean': mse_clean,
            'mse_noisy': mse_noisy,
            'mse_noisy_input': mse_noisy_in,
            'snr_clean': snr_clean,
            'snr_noisy': snr_noisy,
        })
        noisy_denorm = denormalize_mel(noisy_norm.astype(np.float64) / 255.0)
        spec_path = os.path.join(spec_dir, os.path.splitext(wav_name)[0] + '.png')
        save_comparison(clean_denorm, noisy_denorm, rec_clean_denorm, rec_noisy_denorm, spec_path, wav_name)
        # Save MFCC version
        mfcc_path = os.path.join(mfcc_dir, os.path.splitext(wav_name)[0] + '.png')
        try:
            clean_mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=64)
            noisy_mfcc = librosa.feature.mfcc(y=noisy_audio, sr=SAMPLE_RATE, n_mfcc=64)
            save_mfcc_comparison(clean_mfcc, noisy_mfcc, clean_mfcc, noisy_mfcc, mfcc_path, wav_name)
        except:
            pass
        if len(results) >= NUM_SAMPLES:
            break
    return results

def print_and_save(results, attack_name, out_dir, suffix=''):
    avg_mse_c = np.mean([r['mse_clean'] for r in results])
    avg_mse_n = np.mean([r['mse_noisy'] for r in results])
    avg_mse_ni = np.mean([r['mse_noisy_input'] for r in results])
    avg_snr_c = np.mean([r['snr_clean'] for r in results])
    avg_snr_n = np.mean([r['snr_noisy'] for r in results])
    print(f"  Avg MSE (clean rec vs original): {avg_mse_c:.8f}")
    print(f"  Avg MSE (noisy rec vs original): {avg_mse_n:.8f}")
    print(f"  Avg MSE (noisy rec vs noisy in): {avg_mse_ni:.8f}")
    print(f"  Avg SNR (clean rec):             {avg_snr_c:.4f} dB")
    print(f"  Avg SNR (noisy rec):             {avg_snr_n:.4f} dB")
    name = f'results_{attack_name}{suffix}.csv'
    csv_path = os.path.join(out_dir, name)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['sample', 'mse_clean', 'mse_noisy', 'mse_noisy_input', 'snr_clean', 'snr_noisy'])
        w.writeheader()
        w.writerows(results)
    print(f"  Saved -> {csv_path}")
    return avg_mse_n, avg_snr_n

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device(f"cuda:{CUDA_ID}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(CUDA_ID)
    print(f"Device: {device}")
    print("Loading noise...")
    noise_audio = load_noise(NOISE_PATH)
    print(f"  Noise: {len(noise_audio)} samples")
    print("Loading classifier...")
    classifier, ep, acc = load_classifier(BEST_CLASSIFIER_PATH, device)
    print(f"  Classifier: epoch={ep}, acc={acc}")
    print("Loading inversion models...")
    inv_wb, acc_wb = load_inversion(WHITEBOX_INV_PATH, SPLIT_POINT, device)
    inv_bb, acc_bb = load_inversion(BLACKBOX_INV_PATH, SPLIT_POINT, device)
    print(f"  Whitebox:  best_acc={acc_wb}")
    print(f"  Blackbox:  best_acc={acc_bb}")
    print(f"\n{'='*60}")
    print(f"Config: split_point={SPLIT_POINT}, noise_amp={NOISE_AMPLITUDE}, n_samples={NUM_SAMPLES}")
    print(f"{'='*60}")
    all_main = {}
    for name, inv in [('whitebox', inv_wb), ('blackbox', inv_bb)]:
        print(f"\n--- {name.upper()} Attack (clean classifier) ---")
        r = evaluate(classifier, inv, noise_audio, device, tag=name)
        all_main[name] = r
        print_and_save(r, name, OUTPUT_DIR)
    # Summary
    sum_path = os.path.join(OUTPUT_DIR, 'summary.txt')
    with open(sum_path, 'w') as f:
        f.write(f"Noise Evaluation Summary\n{'='*60}\n")
        f.write(f"Noise: {NOISE_PATH}\nAmplitude: {NOISE_AMPLITUDE}\nSplit Point: {SPLIT_POINT}\nSamples: {NUM_SAMPLES}\n\n")
        for name, results in all_main.items():
            avg_mse_n = np.mean([r['mse_noisy'] for r in results])
            avg_snr_n = np.mean([r['snr_noisy'] for r in results])
            f.write(f"{name.upper()}:\n  MSE(noisy)={avg_mse_n:.8f}  SNR(noisy)={avg_snr_n:.4f} dB\n")
    print(f"\nSummary -> {sum_path}")
    # Split Learning Epoch Evaluation
    print(f"\n{'='*60}")
    print("Split Learning Framework - Epoch Evaluation")
    print(f"{'='*60}")
    sl_results = {}
    for epoch_name, ckpt_path in CLASSIFIER_EPOCHS.items():
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] epoch={epoch_name}: {ckpt_path} not found")
            continue
        cl, ep, acc = load_classifier(ckpt_path, device)
        print(f"\n  Epoch {epoch_name} (acc={acc}):")
        sl_results[epoch_name] = {}
        for atk_name, inv in [('whitebox', inv_wb), ('blackbox', inv_bb)]:
            tag = f'sl_ep{epoch_name}_{atk_name}'
            print(f"    {atk_name}:")
            r = evaluate(cl, inv, noise_audio, device, tag=tag)
            sl_results[epoch_name][atk_name] = r
            print_and_save(r, atk_name, OUTPUT_DIR, suffix=f'_ep{epoch_name}')
    sl_path = os.path.join(OUTPUT_DIR, 'split_learning_summary.txt')
    with open(sl_path, 'w') as f:
        f.write(f"Split Learning - Noise Evaluation\n{'='*60}\n\n")
        f.write(f"{'Epoch':<10} {'Attack':<12} {'MSE(noisy)':<20} {'SNR(noisy)':<15}\n")
        f.write(f"{'-'*60}\n")
        for ep in sorted(sl_results.keys()):
            for atk in ['whitebox', 'blackbox']:
                if atk in sl_results[ep]:
                    avg_mse = np.mean([x['mse_noisy'] for x in sl_results[ep][atk]])
                    avg_snr = np.mean([x['snr_noisy'] for x in sl_results[ep][atk]])
                    f.write(f"{ep:<10} {atk:<12} {avg_mse:<20.8f} {avg_snr:<15.4f}\n")
    print(f"\nSplit learning summary -> {sl_path}")
    print(f"\nAll results in {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
