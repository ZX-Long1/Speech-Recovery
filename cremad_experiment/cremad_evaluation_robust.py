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
MAX_TIME = 640
N_FFT = 320
HOP_LENGTH = 80

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
PROCESSED = f'{BASE}/processed'
WAV_DIR = f'{BASE}/AudioWAV'
OUTPUT_DIR = f'{BASE}/eval_results_mel_robust'
WER_SUBSET = 200
AUDIO_SAMPLES = 20
EMO_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}
EMO_INV = {v: k for k, v in EMO_MAP.items()}

MODEL_COMBOS = [
    ('mel', 'whitebox_shallow_robust', [1,2,3,4]),
    ('mel', 'blackbox_shallow_robust',  [1,2,3,4]),
]

NOISE_NAMES = ['clean', 'market', 'office', 'street']

CLS_MAP = {
    'mel':  f'{BASE}/classifier/mel_train_record/classifier.pth',
}

CKPT_DIR_MAP = {
    ('mel', 'whitebox_shallow_robust'): f'{BASE}/inversion/mel/whitebox_shallow_robust',
    ('mel', 'blackbox_shallow_robust'):  f'{BASE}/inversion/mel/blackbox_shallow_robust',
}

INV_CLASS_MAP = {
    ('mel', 'whitebox_shallow_robust'): {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow, 4: Conv_sp4_shallow},
    ('mel', 'blackbox_shallow_robust'):  {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow, 4: Conv_sp4_shallow},
}

def feat_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return _pad_or_trim(mel_db)

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

def save_feature_comparison(clean_feat, noisy_feat, fname, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    titles = ['Clean', 'Noisy']
    imgs = [clean_feat, noisy_feat]
    for ax, img, ti in zip(axes, imgs, titles):
        librosa.display.specshow(img, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                 x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        ax.set_title(ti)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    os.makedirs(f'{OUTPUT_DIR}/spectrograms', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/audio_samples', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/noise_spec', exist_ok=True)

    # Load noise files
    noise_cache = {}
    for n in NOISE_NAMES:
        if n == 'clean':
            noise_cache[n] = None
        else:
            npath = f'/media/sda1/zxlong/tmc_re/noise/{n}.m4a'
            if not os.path.exists(npath):
                npath = npath.replace('.m4a', '.wav')
            noise_cache[n] = load_noise(npath)
            print(f'Loaded noise {n}: {len(noise_cache[n])} samples', flush=True)

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
            'fname': os.path.basename(fp).replace('_mel.npy', ''),
            'wav_path': wav_path,
            'label': label,
        })
    print(f'Test samples: {len(test_samples)}', flush=True)

    random.seed(42)
    wer_indices = set(random.sample(range(len(test_samples)), min(WER_SUBSET, len(test_samples))))
    audio_indices = set(random.sample(list(wer_indices), min(AUDIO_SAMPLES, len(wer_indices))))

    # Whisper
    print('Loading Whisper...', flush=True)
    asr = whisper.load_model('small', device='cuda' if torch.cuda.is_available() else 'cpu')

    # Clean text references (one per sample, independent of condition)
    clean_texts = {}
    for idx in tqdm(wer_indices, desc='Clean transcriptions'):
        samp = test_samples[idx]
        audio, _ = librosa.load(samp['wav_path'], sr=SAMPLE_RATE, mono=True)
        result = asr.transcribe(audio, language='en', task='transcribe', fp16=torch.cuda.is_available())
        clean_texts[idx] = result['text'].strip()

    norm_info = json.load(open(f'{PROCESSED}/norm_info.json'))
    MEL_MIN, MEL_MAX = norm_info['mel_min'], norm_info['mel_max']

    all_rows = []
    wer_rows = []
    noise_spec_saved = {n: 0 for n in NOISE_NAMES if n != 'clean'}

    for feat_type, mode, split_points in MODEL_COMBOS:
        print(f'\n{"="*70}\n{feat_type.upper()} / {mode}\n{"="*70}', flush=True)

        classifier = ResNet18(BasicBlock, 1, 6).to(device)
        cl_path = CLS_MAP[feat_type]
        ckpt = torch.load(cl_path, map_location='cpu')
        classifier.load_state_dict(ckpt['model'])
        classifier.eval()

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
            print(f'  Loaded inversion (best epoch={ckpt.get("best_epoch","?")})', flush=True)

            for cond in NOISE_NAMES:
                print(f'  Condition: {cond}', flush=True)
                mse_list, snr_list = [], []
                corr_inp_list, corr_rec_list = [], []

                for idx, samp in enumerate(tqdm(test_samples, desc=f'eval {mode}/sp{sp}/{cond}')):
                    audio, _ = librosa.load(samp['wav_path'], sr=SAMPLE_RATE, mono=True)
                    label = samp['label']

                    # Condition: noisy or clean audio
                    if cond == 'clean':
                        proc_audio = audio
                    else:
                        proc_audio = add_noise(audio, noise_cache[cond], amp=1.0)

                    feat_db = feat_mel(proc_audio)
                    inp = normalize(feat_db, MEL_MIN, MEL_MAX).unsqueeze(0).to(device)

                    # Save noise feature comparison (first 8 per noise condition)
                    if cond != 'clean' and noise_spec_saved[cond] < 8:
                        clean_feat = feat_mel(audio)
                        save_feature_comparison(
                            clean_feat, feat_db,
                            f'{mode}_sp{sp}_{cond}_{samp["fname"]}.png',
                            f'{OUTPUT_DIR}/noise_spec'
                        )
                        noise_spec_saved[cond] += 1

                    with torch.no_grad():
                        logit_inp = classifier(inp, split_point=0)
                        ft = classifier(inp, split_point=sp)
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
                        'feature': feat_type, 'mode': mode, 'split_point': sp,
                        'condition': cond,
                        'sample': samp['fname'], 'label': EMO_INV[label],
                        'mse': mse, 'snr': snr,
                        'correct_input': corr_inp,
                        'correct_recon': corr_rec,
                    })

                    # ── WER ──
                    if idx in wer_indices:
                        try:
                            recon_db = denormalize(rec, MEL_MIN, MEL_MAX)
                            recon_audio = reconstruct_audio_from_mel(recon_db)
                            result = asr.transcribe(recon_audio, language='en', task='transcribe',
                                                    fp16=torch.cuda.is_available())
                            recon_text = result['text'].strip()
                            clean_text = clean_texts[idx]
                            wer_val = jiwer.wer(clean_text, recon_text)

                            wer_rows.append({
                                'feature': feat_type, 'mode': mode, 'split_point': sp,
                                'condition': cond,
                                'sample': samp['fname'], 'label': EMO_INV[label],
                                'clean_text': clean_text,
                                'recon_text': recon_text,
                                'wer': wer_val,
                            })

                            if idx in audio_indices:
                                sf.write(
                                    f'{OUTPUT_DIR}/audio_samples/{cond}_{samp["fname"]}.wav',
                                    proc_audio, SAMPLE_RATE
                                )
                                sf.write(
                                    f'{OUTPUT_DIR}/audio_samples/recon_{mode}_sp{sp}_{cond}_{samp["fname"]}.wav',
                                    recon_audio, SAMPLE_RATE
                                )
                        except Exception as e:
                            print(f'  WER failed for {samp["fname"]} sp{sp} {cond}: {e}', flush=True)

                    # ── Spectrogram ──
                    if idx % 50 == 0:
                        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        for ax, img, ti in zip(axes, [inp.cpu(), rec.cpu()], ['Original', 'Recon']):
                            show = denormalize(img, MEL_MIN, MEL_MAX)
                            librosa.display.specshow(show, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                                     x_axis='time', y_axis='mel', ax=ax, cmap='magma')
                            ax.set_title(f'{ti} ({mode}/sp{sp}/{cond})')
                        plt.tight_layout()
                        plt.savefig(f'{OUTPUT_DIR}/spectrograms/{mode}_sp{sp}_{cond}_{idx:04d}.png',
                                    dpi=150, bbox_inches='tight')
                        plt.close()

                n = len(test_samples)
                print(f'  {cond}: MSE={np.mean(mse_list):.6f}  SNR={np.mean(snr_list):.2f}  '
                      f'ACC_inp={np.mean(corr_inp_list)*100:.1f}%  ACC_rec={np.mean(corr_rec_list)*100:.1f}%',
                      flush=True)

    # ── Save results ─────────────────────────────────────────────────

    df = pd.DataFrame(all_rows)
    df.to_csv(f'{OUTPUT_DIR}/eval_detail.csv', index=False)
    print(f'\nSaved {OUTPUT_DIR}/eval_detail.csv', flush=True)

    if wer_rows:
        df_wer = pd.DataFrame(wer_rows)
        df_wer.to_csv(f'{OUTPUT_DIR}/wer_detail.csv', index=False)
        print(f'Saved {OUTPUT_DIR}/wer_detail.csv', flush=True)

    summary_rows = []
    for (feat, mode, sp, cond), grp in df.groupby(['feature', 'mode', 'split_point', 'condition']):
        sub_wer = df_wer[(df_wer['feature'] == feat) & (df_wer['mode'] == mode) & 
                         (df_wer['split_point'] == sp) & (df_wer['condition'] == cond)] if 'df_wer' in dir() else pd.DataFrame()
        summary_rows.append({
            'feature': feat, 'mode': mode, 'split_point': sp, 'condition': cond,
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

if __name__ == '__main__':
    main()
