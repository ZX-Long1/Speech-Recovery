import os, sys, random, json
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cremad_experiment.cremad_models import BasicBlock, ResNet18, Conv_sp1, Conv_sp2, Conv_sp3, Conv_sp4

SAMPLE_RATE = 16000
N_MELS = 64
N_MFCC = 64
MAX_TIME = 640
N_FFT = 320
HOP_LENGTH = 80
NC, NZ = 1, 6
NOISE_AMP = 1.0
NUM_SAMPLES = 123
SPLIT_POINTS = [1, 2, 3, 4]

NOISE_PATH = '/media/sda1/zxlong/tmc_re/noise/20260601-0902.m4a'
BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
PROCESSED = f'{BASE}/processed'
AUX_DIR = f'{PROCESSED}/aux'
OUTPUT_DIR = f'{BASE}/noise_eval_results'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f'Device: {DEVICE}', flush=True)

norm_info = json.load(open(f'{PROCESSED}/norm_info.json'))
MEL_MIN = norm_info['mel_min']
MEL_MAX = norm_info['mel_max']
MFCC_MIN = norm_info['mfcc_min']
MFCC_MAX = norm_info['mfcc_max']

INV_CLS = {1: Conv_sp1, 2: Conv_sp2, 3: Conv_sp3, 4: Conv_sp4}

noise_raw, _ = librosa.load(NOISE_PATH, sr=SAMPLE_RATE, mono=True)
noise_power = np.sqrt(np.mean(noise_raw ** 2))

def feat_fn_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
                                          n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel = librosa.power_to_db(mel, ref=np.max)
    return _pad_or_trim(mel)

def feat_fn_mfcc(y):
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    return _pad_or_trim(mfcc)

def _pad_or_trim(feat):
    if feat.shape[1] > MAX_TIME:
        return feat[:, :MAX_TIME]
    import scipy.interpolate as sci
    h, w = feat.shape
    padded = np.zeros((h, MAX_TIME))
    padded[:, :w] = feat
    for i in range(h):
        x_new = np.linspace(0, w-1, MAX_TIME)
        x_old = np.arange(w)
        fi = sci.interp1d(x_old, feat[i], kind='linear', fill_value='extrapolate')
        padded[i] = fi(x_new)
    return padded

def normalize(feat_db, min_val, max_val):
    norm = ((feat_db - min_val) / (max_val - min_val)) * 255
    norm = np.uint8(norm)
    return transforms.ToTensor()(Image.fromarray(norm))

def denormalize(tensor, min_val, max_val):
    arr = tensor.squeeze().cpu().numpy()
    return arr * (max_val - min_val) + min_val

def compute_snr(orig, recon):
    noise_pow = torch.mean((orig - recon) ** 2)
    signal_pow = torch.mean(orig ** 2)
    if noise_pow < 1e-12:
        return float('inf')
    return 10 * torch.log10(signal_pow / noise_pow).item()

aux_files = sorted([f for f in os.listdir(AUX_DIR) if f.endswith('_mel.npy')])
random.seed(42)
selected = random.sample(aux_files, min(NUM_SAMPLES, len(aux_files)))
print(f'Selected {len(selected)} aux samples', flush=True)

audio_cache = {}
for fname in selected:
    wav_name = fname.replace('_mel.npy', '.wav')
    wav_path = os.path.join(f'{BASE}/AudioWAV', wav_name)
    if not os.path.exists(wav_path):
        continue
    data, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
    sig_power = np.sqrt(np.mean(data ** 2))
    scale = NOISE_AMP * sig_power / (noise_power + 1e-12)
    noise_trunc = noise_raw[:len(data)]
    if len(noise_trunc) < len(data):
        noise_trunc = np.tile(noise_raw, int(np.ceil(len(data)/len(noise_raw))))[:len(data)]
    emo = wav_name.split('_')[2]
    label = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}[emo]
    audio_cache[fname] = {'label': label, 'clean': data, 'noisy': data + scale * noise_trunc}

print(f'Loaded {len(audio_cache)} audio files', flush=True)

for feat_type in ['mel', 'mfcc']:
    print(f'\n{"#"*70}\n# {feat_type.upper()}\n{"#"*70}', flush=True)

    if feat_type == 'mel':
        min_val, max_val = MEL_MIN, MEL_MAX
        feat_fn = feat_fn_mel
        spec_type = 'Mel'
    else:
        min_val, max_val = MFCC_MIN, MFCC_MAX
        feat_fn = feat_fn_mfcc
        spec_type = 'MFCC'

    cl_path = f'{BASE}/classifier/{feat_type}_train_record/classifier.pth'
    ckpt = torch.load(cl_path, map_location='cpu')
    classifier = ResNet18(BasicBlock, NC, NZ).to(DEVICE)
    classifier.load_state_dict(ckpt['model'])
    classifier.eval()

    all_rows = []

    for sp in SPLIT_POINTS:
        print(f'\n--- Split Point {sp} ---', flush=True)

        inv_wb = INV_CLS[sp]().to(DEVICE)
        ckpt_wb = torch.load(f'{BASE}/inversion/{feat_type}/whitebox/split_point{sp}/train_record/inversion.pth', map_location='cpu')
        inv_wb.load_state_dict(ckpt_wb['model'])
        inv_wb.eval()

        inv_bb = INV_CLS[sp]().to(DEVICE)
        ckpt_bb = torch.load(f'{BASE}/inversion/{feat_type}/blackbox/split_point{sp}/train_record/inversion.pth', map_location='cpu')
        inv_bb.load_state_dict(ckpt_bb['model'])
        inv_bb.eval()

        for mode, inv_model in [('whitebox', inv_wb), ('blackbox', inv_bb)]:
            mse_c_list, mse_n_list = [], []
            snr_c_list, snr_n_list = [], []
            ci_list, ni_list, cr_list, nr_list = [], [], [], []

            for idx, fname in enumerate(audio_cache):
                ac = audio_cache[fname]
                label = ac['label']

                feat_clean_db = feat_fn(ac['clean'])
                feat_noisy_db = feat_fn(ac['noisy'])

                inp_c = normalize(feat_clean_db, min_val, max_val).unsqueeze(0).to(DEVICE)
                inp_n = normalize(feat_noisy_db, min_val, max_val).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logit_ci = classifier(inp_c, split_point=0)
                    logit_ni = classifier(inp_n, split_point=0)

                    ft_c = classifier(inp_c, split_point=sp)
                    rec_c = inv_model(ft_c)

                    ft_n = classifier(inp_n, split_point=sp)
                    rec_n = inv_model(ft_n)

                    logit_cr = classifier(rec_c, split_point=0)
                    logit_nr = classifier(rec_n, split_point=0)

                ci_list.append(logit_ci.argmax(dim=1).item() == label)
                ni_list.append(logit_ni.argmax(dim=1).item() == label)
                cr_list.append(logit_cr.argmax(dim=1).item() == label)
                nr_list.append(logit_nr.argmax(dim=1).item() == label)

                mse_c = F.mse_loss(rec_c, inp_c).item()
                mse_n = F.mse_loss(rec_n, inp_c).item()
                snr_c = compute_snr(inp_c, rec_c)
                snr_n = compute_snr(inp_c, rec_n)

                mse_c_list.append(mse_c)
                mse_n_list.append(mse_n)
                snr_c_list.append(snr_c)
                snr_n_list.append(snr_n)

                all_rows.append({
                    'feature': feat_type, 'split_point': sp, 'mode': mode,
                    'sample': fname, 'label': label,
                    'mse_clean': mse_c, 'mse_noisy': mse_n,
                    'snr_clean': snr_c, 'snr_noisy': snr_n,
                    'correct_clean_inp': int(ci_list[-1]),
                    'correct_noisy_inp': int(ni_list[-1]),
                    'correct_clean_rec': int(cr_list[-1]),
                    'correct_noisy_rec': int(nr_list[-1]),
                })

                if idx % 5 == 0:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    for ax, ti, img in zip(axes.flat,
                        ['Original Clean', 'Noisy Input', 'Clean Recon', 'Noisy Recon'],
                        [inp_c, inp_n, rec_c, rec_n]):
                        show = denormalize(img, min_val, max_val)
                        y_axis = 'mel' if feat_type == 'mel' else 'linear'
                        librosa.display.specshow(show, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                                 x_axis='time', y_axis=y_axis, ax=ax, cmap='magma')
                        ax.set_title(f'{ti}\n(sp={sp}, {mode})')
                    plt.tight_layout()
                    d = os.path.join(OUTPUT_DIR, f'spectrograms_{feat_type}', mode, f'sp{sp}')
                    os.makedirs(d, exist_ok=True)
                    plt.savefig(os.path.join(d, f'{idx:03d}.png'), dpi=150, bbox_inches='tight')
                    plt.close()

            n = len(audio_cache)
            acc_ci = np.mean(ci_list) * 100
            acc_ni = np.mean(ni_list) * 100
            acc_cr = np.mean(cr_list) * 100
            acc_nr = np.mean(nr_list) * 100
            print(f'  {mode:>8}: MSE_c={np.mean(mse_c_list):.6f} MSE_n={np.mean(mse_n_list):.6f}  '
                  f'SNR_c={np.mean(snr_c_list):.2f} SNR_n={np.mean(snr_n_list):.2f}  '
                  f'Acc_ci={acc_ci:.1f} Acc_ni={acc_ni:.1f} Acc_cr={acc_cr:.1f} Acc_nr={acc_nr:.1f}', flush=True)

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_DIR, f'results_{feat_type}.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved {csv_path}', flush=True)

    summary_path = os.path.join(OUTPUT_DIR, f'summary_{feat_type}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"CREMA-D Noise Evaluation - {spec_type}\n")
        f.write(f"Noise: noise_amp={NOISE_AMP}, samples={len(audio_cache)}\n\n")
        hdr = f"{'Split':<6} {'Mode':<10} {'MSE_c':<10} {'MSE_n':<10} {'SNR_c':<9} {'SNR_n':<9} {'Acc_ci':<8} {'Acc_ni':<8} {'Acc_cr':<8} {'Acc_nr':<8}"
        f.write(hdr + "\n" + "-" * 86 + "\n")
        for sp in SPLIT_POINTS:
            for mode in ['whitebox', 'blackbox']:
                sub = df[(df['split_point'] == sp) & (df['mode'] == mode)]
                mc = sub['mse_clean'].mean()
                mn = sub['mse_noisy'].mean()
                sc = sub['snr_clean'].mean()
                sn = sub['snr_noisy'].mean()
                ci = sub['correct_clean_inp'].mean() * 100
                ni = sub['correct_noisy_inp'].mean() * 100
                cr = sub['correct_clean_rec'].mean() * 100
                nr = sub['correct_noisy_rec'].mean() * 100
                f.write(f"{sp:<6} {mode:<10} {mc:<10.6f} {mn:<10.6f} {sc:<9.2f} {sn:<9.2f} {ci:<8.1f} {ni:<8.1f} {cr:<8.1f} {nr:<8.1f}\n")
    print(f'Saved {summary_path}', flush=True)

print('\nDone!', flush=True)
