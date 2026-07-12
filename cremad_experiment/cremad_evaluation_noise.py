import os, sys, json, random, warnings, argparse
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
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cremad_experiment.cremad_models import (
    BasicBlock, ResNet18,
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
OUTPUT_ROOT = f'{BASE}/eval_results_noise'
WER_SUBSET = 200
AUDIO_SAMPLES = 20
EMO_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
EMO_INV = {v: k for k, v in EMO_MAP.items()}

MODEL_COMBOS = [
    ('mel', 'whitebox_shallow', [1,2,3,4]),
    ('mel', 'blackbox',         [1,2,3,4]),
    ('mfcc', 'whitebox_shallow',  [1,2,3,4]),
    ('mfcc', 'blackbox_shallow',  [1,2,3,4]),
]

CLS_MAP = {
    'mel':  f'{BASE}/classifier/mel_train_record/classifier.pth',
    'mfcc': f'{BASE}/classifier/mfcc_train_record/classifier.pth',
}

CKPT_DIR_MAP = {
    ('mel', 'whitebox_shallow'): f'{BASE}/inversion/mel/whitebox_shallow',
    ('mel', 'blackbox'):         f'{BASE}/inversion/mel/blackbox',
    ('mfcc', 'whitebox_shallow'):  f'{BASE}/inversion/mfcc/whitebox_shallow',
    ('mfcc', 'blackbox_shallow'):  f'{BASE}/inversion/mfcc/blackbox_shallow',
}

INV_CLASS_MAP = {
    ('mel', 'whitebox_shallow'): {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow, 4: Conv_sp4_shallow},
    ('mel', 'blackbox'):         {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow, 4: Conv_sp4_shallow},
    ('mfcc', 'whitebox_shallow'):  {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow, 4: Conv_sp4_shallow},
    ('mfcc', 'blackbox_shallow'):  {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow, 4: Conv_sp4_shallow},
}

# ── Feature helpers ──────────────────────────────────────────────────

def feat_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return _pad_or_trim(mel_db)

def feat_mfcc(y):
    m = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
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
    return tensor.squeeze().cpu().numpy() * (mx - mn) + mn

def compute_snr(orig, recon):
    noise_pow = torch.mean((orig - recon) ** 2)
    signal_pow = torch.mean(orig ** 2)
    if noise_pow < 1e-12:
        return float('inf')
    return 10 * torch.log10(signal_pow / noise_pow).item()

def reconstruct_audio_from_mel(mel_db_np, sr=SAMPLE_RATE, n_fft=N_FFT, hop=HOP_LENGTH):
    mel_power = librosa.db_to_power(mel_db_np)
    mel_power = np.clip(mel_power, 1e-10, None)
    audio = librosa.feature.inverse.mel_to_audio(mel_power, sr=sr, n_fft=n_fft, hop_length=hop, power=2.0, n_iter=32)
    return np.nan_to_num(audio, nan=0.0)

def reconstruct_audio_from_mfcc(mfcc_np, sr=SAMPLE_RATE, n_fft=N_FFT, hop=HOP_LENGTH, n_mels=N_MELS):
    mel_power = librosa.feature.inverse.mfcc_to_mel(mfcc_np, n_mels=n_mels)
    mel_power = np.nan_to_num(mel_power, nan=0.0, posinf=0.0, neginf=0.0)
    mel_power = np.maximum(mel_power, 1e-10)
    try:
        audio = librosa.feature.inverse.mel_to_audio(mel_power, sr=sr, n_fft=n_fft, hop_length=hop, power=2.0, n_iter=32)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio
    except:
        return None

# ── Noise helpers ────────────────────────────────────────────────────

def load_noise(path, target_sr=SAMPLE_RATE):
    noise, sr = librosa.load(path, sr=None, mono=True)
    if sr != target_sr:
        noise = librosa.resample(noise, orig_sr=sr, target_sr=target_sr)
    if np.max(np.abs(noise)) > 0:
        noise = noise / np.max(np.abs(noise))
    return noise

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

# ── Plot helpers ─────────────────────────────────────────────────────

def save_feature_comparison(clean_feat, noisy_feat, fname, save_dir, feat_type):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    titles = ['Clean', f'Noisy ({feat_type})']
    imgs = [clean_feat, noisy_feat]
    y_axis = 'mel' if feat_type == 'mel' else 'linear'
    for ax, img, ti in zip(axes, imgs, titles):
        librosa.display.specshow(img, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                 x_axis='time', y_axis=y_axis, ax=ax, cmap='magma')
        ax.set_title(ti)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', type=str, required=True, choices=['market', 'office', 'street'])
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    noise_name = args.noise
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    OUTPUT_DIR = f'{OUTPUT_ROOT}/{noise_name}'
    os.makedirs(f'{OUTPUT_DIR}/spectrograms', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/audio_samples', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/noise_spec', exist_ok=True)

    # Load noise
    noise_path = f'/media/sda1/zxlong/tmc_re/noise/{noise_name}.m4a'
    if not os.path.exists(noise_path):
        noise_path = noise_path.replace('.m4a', '.wav')
    print(f'Loading noise: {noise_path}', flush=True)
    noise_audio = load_noise(noise_path)
    print(f'  Noise: {len(noise_audio)} samples ({len(noise_audio)/SAMPLE_RATE:.2f}s)', flush=True)

    # Load test samples (same as eval_results)
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

    random.seed(42)
    wer_indices = set(random.sample(range(len(test_samples)), min(WER_SUBSET, len(test_samples))))
    audio_indices = set(random.sample(list(wer_indices), min(AUDIO_SAMPLES, len(wer_indices))))

    # Load Whisper
    print('Loading Whisper...', flush=True)
    asr = whisper.load_model('small', device='cuda' if torch.cuda.is_available() else 'cpu')

    # Transcribe clean audio (WER subset reference)
    clean_texts = {}
    for idx in tqdm(wer_indices, desc='Clean transcriptions'):
        samp = test_samples[idx]
        audio, _ = librosa.load(samp['wav_path'], sr=SAMPLE_RATE, mono=True)
        result = asr.transcribe(audio, language='en', task='transcribe', fp16=torch.cuda.is_available())
        clean_texts[idx] = result['text'].strip()

    # Normalization info
    norm_info = json.load(open(f'{PROCESSED}/norm_info.json'))
    MEL_MIN, MEL_MAX = norm_info['mel_min'], norm_info['mel_max']
    MFCC_MIN, MFCC_MAX = norm_info['mfcc_min'], norm_info['mfcc_max']

    all_rows = []
    wer_rows = []

    # Save some noise feature visualizations
    noise_spec_saved = False

    for feat_type, mode, split_points in MODEL_COMBOS:
        print(f'\n{"="*70}\n{feat_type.upper()} / {mode}\n{"="*70}', flush=True)

        classifier = ResNet18(BasicBlock, 1, 6).to(device)
        cl_path = CLS_MAP[feat_type]
        ckpt = torch.load(cl_path, map_location='cpu')
        classifier.load_state_dict(ckpt['model'])
        classifier.eval()

        mn, mx = (MEL_MIN, MEL_MAX) if feat_type == 'mel' else (MFCC_MIN, MFCC_MAX)
        feat_fn = feat_mel if feat_type == 'mel' else feat_mfcc
        inv_cls_map = INV_CLASS_MAP[(feat_type, mode)]
        ckpt_dir = CKPT_DIR_MAP[(feat_type, mode)]

        for sp in split_points:
            print(f'\n--- Split Point {sp} ---', flush=True)

            inversion = inv_cls_map[sp]().to(device)
            ckpt_path = f'{ckpt_dir}/split_point{sp}/train_record/inversion.pth'
            if not os.path.exists(ckpt_path):
                print(f'  WARNING: checkpoint not found at {ckpt_path}, skipping', flush=True)
                continue
            ckpt = torch.load(ckpt_path, map_location='cpu')
            inversion.load_state_dict(ckpt['model'])
            inversion.eval()

            mse_clean_list, mse_noisy_list = [], []
            snr_clean_list, snr_noisy_list = [], []
            corr_inp_list, corr_rec_clean_list, corr_rec_noisy_list = [], [], []

            n_noise_spec = 0

            for idx, samp in enumerate(tqdm(test_samples, desc=f'eval {feat_type}/{mode}/sp{sp}')):
                audio, _ = librosa.load(samp['wav_path'], sr=SAMPLE_RATE, mono=True)
                label = samp['label']

                # Clean (baseline)
                feat_db_clean = feat_fn(audio)
                inp_clean = normalize(feat_db_clean, mn, mx).unsqueeze(0).to(device)

                # Noisy
                noisy_audio = add_noise(audio, noise_audio, amp=1.0)
                feat_db_noisy = feat_fn(noisy_audio)
                inp_noisy = normalize(feat_db_noisy, mn, mx).unsqueeze(0).to(device)

                # Save noise feature comparison images (first 8 samples per model combo)
                if n_noise_spec < 8:
                    save_feature_comparison(
                        feat_db_clean, feat_db_noisy,
                        f'{feat_type}_{mode}_sp{sp}_{samp["fname"]}.png',
                        f'{OUTPUT_DIR}/noise_spec', feat_type
                    )
                    n_noise_spec += 1

                with torch.no_grad():
                    # Classifier logits from clean input
                    logit_inp = classifier(inp_clean, split_point=0)

                    # Clean reconstruction
                    ft_clean = classifier(inp_clean, split_point=sp)
                    rec_clean = inversion(ft_clean)
                    logit_rec_clean = classifier(rec_clean, split_point=0)

                    # Noisy reconstruction
                    ft_noisy = classifier(inp_noisy, split_point=sp)
                    rec_noisy = inversion(ft_noisy)
                    logit_rec_noisy = classifier(rec_noisy, split_point=0)

                corr_inp = int(logit_inp.argmax(dim=1).item() == label)
                corr_rec_clean = int(logit_rec_clean.argmax(dim=1).item() == label)
                corr_rec_noisy = int(logit_rec_noisy.argmax(dim=1).item() == label)

                mse_clean = F.mse_loss(rec_clean, inp_clean).item()
                mse_noisy = F.mse_loss(rec_noisy, inp_clean).item()
                snr_clean = compute_snr(inp_clean, rec_clean)
                snr_noisy = compute_snr(inp_clean, rec_noisy)

                mse_clean_list.append(mse_clean)
                mse_noisy_list.append(mse_noisy)
                snr_clean_list.append(snr_clean)
                snr_noisy_list.append(snr_noisy)
                corr_inp_list.append(corr_inp)
                corr_rec_clean_list.append(corr_rec_clean)
                corr_rec_noisy_list.append(corr_rec_noisy)

                all_rows.append({
                    'feature': feat_type, 'mode': mode, 'split_point': sp,
                    'noise': noise_name,
                    'sample': samp['fname'], 'label': EMO_INV[label],
                    'mse_clean': mse_clean, 'mse_noisy': mse_noisy,
                    'snr_clean': snr_clean, 'snr_noisy': snr_noisy,
                    'correct_input': corr_inp,
                    'correct_recon_clean': corr_rec_clean,
                    'correct_recon_noisy': corr_rec_noisy,
                })

                # ── WER (noisy reconstruction) ──
                if idx in wer_indices:
                    try:
                        if feat_type == 'mel':
                            recon_db = denormalize(rec_noisy, mn, mx)
                            recon_audio = reconstruct_audio_from_mel(recon_db)
                        else:
                            recon_mfcc = denormalize(rec_noisy, mn, mx)
                            recon_audio = reconstruct_audio_from_mfcc(recon_mfcc)
                        if recon_audio is None:
                            continue
                        result = asr.transcribe(recon_audio, language='en', task='transcribe',
                                                fp16=torch.cuda.is_available())
                        recon_text = result['text'].strip()
                        clean_text = clean_texts[idx]
                        wer_val = jiwer.wer(clean_text, recon_text)

                        wer_rows.append({
                            'feature': feat_type, 'mode': mode, 'split_point': sp,
                            'noise': noise_name,
                            'sample': samp['fname'], 'label': EMO_INV[label],
                            'clean_text': clean_text,
                            'recon_text': recon_text,
                            'wer': wer_val,
                        })

                        if idx in audio_indices:
                            sf.write(
                                f'{OUTPUT_DIR}/audio_samples/clean_{samp["fname"]}.wav',
                                audio, SAMPLE_RATE
                            )
                            sf.write(
                                f'{OUTPUT_DIR}/audio_samples/noisy_{feat_type}_{mode}_sp{sp}_{samp["fname"]}.wav',
                                recon_audio, SAMPLE_RATE
                            )
                    except Exception as e:
                        print(f'  WER failed for {samp["fname"]} sp{sp}: {e}', flush=True)

                # ── Spectrogram (every 50th) ──
                if idx % 50 == 0:
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    titles = ['Original (clean)', 'Recon from clean',
                              'Noisy Input', 'Recon from noisy']
                    imgs = [inp_clean.cpu(), rec_clean.cpu(),
                            inp_noisy.cpu(), rec_noisy.cpu()]
                    for ax, img, ti in zip(axes.flatten(), imgs, titles):
                        show = denormalize(img, mn, mx)
                        y_axis = 'mel' if feat_type == 'mel' else 'linear'
                        librosa.display.specshow(show, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                                 x_axis='time', y_axis=y_axis, ax=ax, cmap='magma')
                        ax.set_title(f'{ti} ({feat_type}/{mode}/sp{sp})')
                    plt.tight_layout()
                    plt.savefig(f'{OUTPUT_DIR}/spectrograms/{feat_type}_{mode}_sp{sp}_{idx:04d}.png',
                                dpi=150, bbox_inches='tight')
                    plt.close()

            n = len(test_samples)
            print(f'  Clean: MSE={np.mean(mse_clean_list):.6f}  SNR={np.mean(snr_clean_list):.2f}  '
                  f'ACC_rec={np.mean(corr_rec_clean_list)*100:.1f}%', flush=True)
            print(f'  Noisy: MSE={np.mean(mse_noisy_list):.6f}  SNR={np.mean(snr_noisy_list):.2f}  '
                  f'ACC_rec={np.mean(corr_rec_noisy_list)*100:.1f}%', flush=True)

    # ── Save results ─────────────────────────────────────────────────

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
            'noise': noise_name,
            'mse_clean_mean': grp['mse_clean'].mean(), 'mse_clean_std': grp['mse_clean'].std(),
            'mse_noisy_mean': grp['mse_noisy'].mean(), 'mse_noisy_std': grp['mse_noisy'].std(),
            'snr_clean_mean': grp['snr_clean'].mean(), 'snr_clean_std': grp['snr_clean'].std(),
            'snr_noisy_mean': grp['snr_noisy'].mean(), 'snr_noisy_std': grp['snr_noisy'].std(),
            'acc_input': grp['correct_input'].mean() * 100,
            'acc_recon_clean': grp['correct_recon_clean'].mean() * 100,
            'acc_recon_noisy': grp['correct_recon_noisy'].mean() * 100,
            'acc_drop_clean': (grp['correct_input'].mean() - grp['correct_recon_clean'].mean()) * 100,
            'acc_drop_noisy': (grp['correct_input'].mean() - grp['correct_recon_noisy'].mean()) * 100,
            'wer_mean': sub_wer['wer'].mean() if len(sub_wer) > 0 else None,
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f'{OUTPUT_DIR}/eval_summary.csv', index=False)
    print(f'Saved {OUTPUT_DIR}/eval_summary.csv', flush=True)

    print('\n' + '=' * 70)
    print(df_summary.to_string(index=False))
    print('=' * 70)
    print(f'\nDone! Noise={noise_name}', flush=True)


if __name__ == '__main__':
    main()
