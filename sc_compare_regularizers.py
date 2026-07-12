import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

SC_BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'
SAMPLE_RATE = 49050
TEST_WAV = f'{SC_BASE}/audio/test_set'
N_MELS = 64
MAX_COLUMN = 96
FRAME_LENGTH = 0.02
N_FFT = int(FRAME_LENGTH * SAMPLE_RATE)
HOP_LENGTH = N_FFT // 4
MEL_MIN_VAL = -80.00000381469727
MEL_MAX_VAL = 3.814697265625e-06

OUTPUT_DIR = 'sc_regularizer_comparison'
os.makedirs(f'{OUTPUT_DIR}/mel_vis', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/audio', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/psd', exist_ok=True)

LAMBDAS = [0.0, 0.5, 1.0, 1.5, 2.0]
SPLIT_POINTS = [1, 2, 3, 4]
COLORS = {'none': '#888888', 'tvd': '#21918c', 'speech_tf': '#fde725'}
LINESTYLES = {'none': '--', 'tvd': '--', 'speech_tf': '-'}

def feat_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] > MAX_COLUMN:
        return mel_db[:, :MAX_COLUMN]
    import scipy.interpolate as sci
    h, w = mel_db.shape
    padded = np.zeros((h, MAX_COLUMN))
    padded[:, :w] = mel_db
    for i in range(h):
        x_new = np.linspace(0, w - 1, MAX_COLUMN)
        x_old = np.arange(w)
        fi = sci.interp1d(x_old, mel_db[i], kind='linear', fill_value='extrapolate')
        padded[i] = fi(x_new)
    return padded

def normalize(feat_db):
    norm = ((feat_db - MEL_MIN_VAL) / (MEL_MAX_VAL - MEL_MIN_VAL)) * 255
    return torch.from_numpy(np.uint8(norm)).unsqueeze(0).float() / 255.0

def denormalize(tensor):
    return tensor.squeeze().cpu().numpy() * (MEL_MAX_VAL - MEL_MIN_VAL) + MEL_MIN_VAL

# 1. Read evaluation summaries
def read_summary(reg_tag, ntv):
    candidates = [f'sc_eval_results_mel_nTV{ntv}']
    if reg_tag != 'none':
        candidates.append(f'sc_eval_results_mel_{reg_tag}_nTV{ntv}')
    for prefix in candidates:
        path = f'{prefix}/eval_summary.csv'
        if os.path.exists(path):
            return pd.read_csv(path)
    return None

# Build comparison table
print("Building comparison table...")
rows = []
best_tvd_lam = {}  # per-split-point best TVD λ
best_stf_lam = {}  # per-split-point best speech_tf λ
for sp in SPLIT_POINTS:
    best_tvd = {'mse': float('inf'), 'ntv': None}
    best_stf = {'mse': float('inf'), 'ntv': None}

    # none: nTV=0.0
    df_none = read_summary('none', 0.0)
    none_row = df_none[(df_none['split_point'] == sp) & (df_none['mode'] == 'whitebox_shallow')] if df_none is not None else pd.DataFrame()

    # tvd: find best lambda
    for lam in LAMBDAS:
        df_tvd = read_summary('tvd' if lam != 0.0 else 'none', lam)
        if df_tvd is not None:
            r = df_tvd[(df_tvd['split_point'] == sp) & (df_tvd['mode'] == 'whitebox_shallow')]
            if len(r) > 0 and r['mse_mean'].values[0] < best_tvd['mse']:
                best_tvd['mse'] = r['mse_mean'].values[0]
                best_tvd['ntv'] = lam
                best_tvd['row'] = r

    for row_set, tag in [(none_row, 'none'), (best_tvd.get('row'), 'tvd')]:
        if row_set is not None and len(row_set) > 0:
            r = row_set.iloc[0]
            rows.append({
                'split_point': sp, 'regularizer': tag,
                'lambda': 0.0 if tag == 'none' else best_tvd['ntv'],
                'mse': r['mse_mean'], 'snr': r['snr_mean'],
                'acc_input': r['acc_input'], 'acc_recon': r['acc_recon'],
                'acc_drop': r['acc_drop'],
            })

    best_tvd_lam[sp] = best_tvd['ntv'] if best_tvd['ntv'] is not None else 0.0

# speech_tf: find best lambda (from existing eval if available, else from speech_tf directories)
for sp in SPLIT_POINTS:
    best_stf = {'mse': float('inf'), 'ntv': None}
    for lam in LAMBDAS:
        path = f'sc_eval_results_mel_speech_tf_nTV{lam}/eval_summary.csv'
        if os.path.exists(path):
            df = pd.read_csv(path)
            r = df[(df['split_point'] == sp) & (df['mode'] == 'whitebox_shallow')]
            if len(r) > 0 and r['mse_mean'].values[0] < best_stf['mse']:
                best_stf['mse'] = r['mse_mean'].values[0]
                best_stf['ntv'] = lam
    if best_stf['ntv'] is not None:
        df = pd.read_csv(f'sc_eval_results_mel_speech_tf_nTV{best_stf["ntv"]}/eval_summary.csv')
        r = df[(df['split_point'] == sp) & (df['mode'] == 'whitebox_shallow')].iloc[0]
        rows.append({
            'split_point': sp, 'regularizer': 'speech_tf',
            'lambda': best_stf['ntv'],
            'mse': r['mse_mean'], 'snr': r['snr_mean'],
            'acc_input': r['acc_input'], 'acc_recon': r['acc_recon'],
            'acc_drop': r['acc_drop'],
        })

    best_stf_lam[sp] = best_stf['ntv'] if best_stf['ntv'] is not None else 0.0

df_summary = pd.DataFrame(rows)
df_summary.to_csv(f'{OUTPUT_DIR}/comparison_summary.csv', index=False)
print(f"Saved {OUTPUT_DIR}/comparison_summary.csv")
print(df_summary.to_string(index=False))

# 2. Mel visualization (load first 5 test samples)
print("\nGenerating mel visualizations...")
import torch
wav_files = sorted(os.listdir(TEST_WAV))
selected_wavs = [w for w in wav_files if int(w.split('_')[0]) in [0, 1, 2, 3, 4]][:5]

for sp in SPLIT_POINTS:
    for wav_name in selected_wavs:
        wav_path = os.path.join(TEST_WAV, wav_name)
        audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=False)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)

        # Original mel
        inp_db = feat_mel(audio)

        fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
        axes[0].imshow(inp_db, aspect='auto', origin='lower', cmap='magma',
                       extent=[0, MAX_COLUMN, 0, N_MELS])
        axes[0].set_title(f'Original ({wav_name.split("_")[0]})')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Mel band')

        for idx, (reg_tag, lam) in enumerate([('none', 0.0), ('tvd', None), ('speech_tf', None)]):
            if reg_tag == 'none':
                npy_path = f'sc_eval_results_mel_nTV0.0/reconstructed_mel/whitebox_shallow/sp{sp}/{wav_name.replace(".wav","")}.npy'
            elif reg_tag == 'tvd':
                # Find best TVD .npy from any lambda that has this sample
                npy_path = None
                for lam_try in LAMBDAS:
                    p = f'sc_eval_results_mel_nTV{lam_try}/reconstructed_mel/whitebox_shallow/sp{sp}/{wav_name.replace(".wav","")}.npy'
                    if os.path.exists(p):
                        npy_path = p
                        break
            else:
                npy_path = f'sc_eval_results_mel_speech_tf_nTV{best_stf["ntv"]}/reconstructed_mel/whitebox_shallow/sp{sp}/{wav_name.replace(".wav","")}.npy'

            if npy_path and os.path.exists(npy_path):
                rec_db = np.load(npy_path)
                axes[idx+1].imshow(rec_db, aspect='auto', origin='lower', cmap='magma',
                                   extent=[0, MAX_COLUMN, 0, N_MELS])
                axes[idx+1].set_title(f'{reg_tag} (λ={best_stf["ntv"] if reg_tag == "speech_tf" else (0.0 if reg_tag == "none" else "best")})')
            else:
                axes[idx+1].text(0.5, 0.5, 'N/A', ha='center', va='center', transform=axes[idx+1].transAxes)
            axes[idx+1].set_xlabel('Time')

        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/mel_vis/sp{sp}_{wav_name.replace(".wav","")}.png', dpi=150, bbox_inches='tight')
        plt.close()

print(f"Saved mel visualizations to {OUTPUT_DIR}/mel_vis/")

# 3. PSD comparison (aggregated over N_SAMPLES, pseudo-inverse mel→PSD)
print("\nGenerating PSD comparison...")

# Pre-compute mel filterbank pseudo-inverse
M = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
M_pinv = np.linalg.pinv(M)
freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)

def mel_to_psd(mel_db_np):
    power_mel = 10.0 ** (mel_db_np / 10.0)
    linear_power = M_pinv @ power_mel
    linear_power = np.maximum(linear_power, 1e-12)
    psd = np.mean(linear_power, axis=1)
    return 10.0 * np.log10(psd)

# Find common samples across all regularizers and λ for whitebox_shallow
mode = 'whitebox_shallow'
sample_sets = []
for lam in LAMBDAS:
    for reg_prefix in ['nTV', 'speech_tf_nTV']:
        d = f'sc_eval_results_mel_{reg_prefix}{lam}/reconstructed_mel/{mode}/sp1'
        if os.path.isdir(d):
            files = [f.replace('.npy', '') for f in os.listdir(d) if f.endswith('.npy')]
            sample_sets.append(set(files))
common = set.intersection(*sample_sets) if all(s for s in sample_sets) else set()
if len(common) == 0:
    common = set(os.listdir(TEST_MEL_NPY)).union(*(s for s in sample_sets))
selected = sorted(common)[:30]
print(f"  Common samples for PSD: {len(common)}, using {len(selected)}")

# Pre-load mel cache for input PSD computation
mel_cache = torch.load('/tmp/sc_mel_cache/test_mel_dict.pt', map_location='cpu')
print(f'  Loaded mel cache: {len(mel_cache)} files', flush=True)

# Colormap for λ — matching VT's magma colormap
VT_COLORS = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
LAM_COLORMAP = {lam: VT_COLORS[i] for i, lam in enumerate(LAMBDAS)}

def get_npy_path(reg_type, lam, sp, sample):
    if reg_type == 'none':
        return f'sc_eval_results_mel_nTV{lam}/reconstructed_mel/{mode}/sp{sp}/{sample}.npy'
    elif reg_type == 'tvd':
        for lam_try in [lam] + [l for l in LAMBDAS if l != lam]:
            p = f'sc_eval_results_mel_nTV{lam_try}/reconstructed_mel/{mode}/sp{sp}/{sample}.npy'
            if os.path.exists(p):
                return p
    elif reg_type == 'speech_tf':
        return f'sc_eval_results_mel_speech_tf_nTV{lam}/reconstructed_mel/{mode}/sp{sp}/{sample}.npy'
    return None

def compute_psds(reg_type, lam, sp, selected):
    """Compute PSDs for all selected samples; return list of psd arrays."""
    psds = []
    for sname in selected:
        p = get_npy_path(reg_type, lam, sp, sname)
        if p and os.path.exists(p):
            mel_db = np.load(p)
            psds.append(mel_to_psd(mel_db))
    return psds if psds else None

all_rows = []
for sp in SPLIT_POINTS:
    # Input PSD (from mel_cache or WAV)
    inp_psds = []
    for sname in selected:
        wav_name = sname + '.wav'
        if wav_name in mel_cache:
            mel_db = mel_cache[wav_name]
        else:
            wav_path = os.path.join(TEST_WAV, wav_name)
            if os.path.exists(wav_path):
                y, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=False)
                if y.ndim > 1:
                    y = np.mean(y, axis=0)
                mel_db = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LENGTH)
                mel_db = librosa.power_to_db(mel_db, ref=np.max)
            else:
                continue
        inp_psds.append(mel_to_psd(mel_db))
    if len(inp_psds) == 0:
        continue
    inp_mean = np.mean(inp_psds, axis=0)

    # Collect PSD per (reg_type, lam)
    configs = [('none', 0.0)] + \
              [('tvd', lam) for lam in [0.5, 1.0, 1.5, 2.0]] + \
              [('speech_tf', lam) for lam in [0.5, 1.0, 1.5, 2.0]]
    all_psds = {}
    for reg_type, lam in configs:
        psds = compute_psds(reg_type, lam, sp, selected)
        if psds is not None:
            all_psds[(reg_type, lam)] = {
                'mean': np.mean(psds, axis=0),
                'std': np.std(psds, axis=0),
            }

    # Dual-panel figures — one per regularizer + none (matching VT style)
    for focus_reg, focus_label in [('tvd', 'TVD'), ('speech_tf', 'STF')]:
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax_top.plot(freqs / 1000, inp_mean, 'k-', linewidth=2.5, label='Input', alpha=0.7, zorder=5)

        # none (λ=0.0)
        if ('none', 0.0) in all_psds:
            r = all_psds[('none', 0.0)]
            ax_top.plot(freqs / 1000, r['mean'], color=VT_COLORS[0], linewidth=1.8, label=f'none (λ=0.0)')

        # Focus regularizer with all λ
        for lam in [0.5, 1.0, 1.5, 2.0]:
            key = (focus_reg, lam)
            if key not in all_psds:
                continue
            r = all_psds[key]
            ax_top.plot(freqs / 1000, r['mean'], color=LAM_COLORMAP[lam], linewidth=1.8,
                       label=f'{focus_label} λ={lam}')

        ax_top.set_ylabel('PSD (dB)', fontsize=12)
        ax_top.set_title(f'{focus_label} + none — Split Point {sp}', fontsize=13)
        ax_top.legend(fontsize=9, loc='lower left')
        ax_top.grid(True, alpha=0.2)
        ax_top.set_xlim(0, 16)

        # Lower: ΔPSD
        if ('none', 0.0) in all_psds:
            delta = all_psds[('none', 0.0)]['mean'] - inp_mean
            ax_bot.plot(freqs / 1000, delta, color=VT_COLORS[0], linewidth=1.8, label='none (λ=0.0)')
        for lam in [0.5, 1.0, 1.5, 2.0]:
            key = (focus_reg, lam)
            if key not in all_psds:
                continue
            delta = all_psds[key]['mean'] - inp_mean
            ax_bot.plot(freqs / 1000, delta, color=LAM_COLORMAP[lam], linewidth=1.8, label=f'{focus_label} λ={lam}')

        ax_bot.axhline(y=0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
        for thresh in [-3, 3]:
            ax_bot.axhline(y=thresh, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)
        ax_bot.set_xlabel('Frequency (kHz)', fontsize=12)
        ax_bot.set_ylabel('ΔPSD (dB)', fontsize=12)
        ax_bot.legend(fontsize=9)
        ax_bot.grid(True, alpha=0.2)
        ax_bot.set_xlim(0, 16)

        fig.tight_layout()
        fname = f'dpsd_ref_{focus_reg}_sp{sp}.png'
        fig.savefig(f'{OUTPUT_DIR}/psd/{fname}', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  {fname}")

    # Combined figure: none + all TVD + all STF
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax_top.plot(freqs / 1000, inp_mean, 'k-', linewidth=2.5, label='Input', alpha=0.7, zorder=5)
    if ('none', 0.0) in all_psds:
        r = all_psds[('none', 0.0)]
        ax_top.plot(freqs / 1000, r['mean'], color=VT_COLORS[0], linewidth=1.8, label='none (λ=0.0)')
    for reg_type, reg_label in [('tvd', 'TVD'), ('speech_tf', 'STF')]:
        for lam in [0.5, 1.0, 1.5, 2.0]:
            key = (reg_type, lam)
            if key not in all_psds:
                continue
            r = all_psds[key]
            ls = '-' if reg_type == 'tvd' else '--'
            ax_top.plot(freqs / 1000, r['mean'], color=LAM_COLORMAP[lam], linewidth=1.5,
                       ls=ls, label=f'{reg_label} λ={lam}')
    ax_top.set_ylabel('PSD (dB)', fontsize=12)
    ax_top.set_title(f'All Regularizers — Split Point {sp}', fontsize=13)
    ax_top.legend(fontsize=8, loc='lower left')
    ax_top.grid(True, alpha=0.2)
    ax_top.set_xlim(0, 16)

    if ('none', 0.0) in all_psds:
        delta = all_psds[('none', 0.0)]['mean'] - inp_mean
        ax_bot.plot(freqs / 1000, delta, color=VT_COLORS[0], linewidth=1.8, label='none (λ=0.0)')
    for reg_type, reg_label in [('tvd', 'TVD'), ('speech_tf', 'STF')]:
        for lam in [0.5, 1.0, 1.5, 2.0]:
            key = (reg_type, lam)
            if key not in all_psds:
                continue
            delta = all_psds[key]['mean'] - inp_mean
            ls = '-' if reg_type == 'tvd' else '--'
            ax_bot.plot(freqs / 1000, delta, color=LAM_COLORMAP[lam], linewidth=1.5,
                       ls=ls, label=f'{reg_label} λ={lam}')
    ax_bot.axhline(y=0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
    for thresh in [-3, 3]:
        ax_bot.axhline(y=thresh, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)
    ax_bot.set_xlabel('Frequency (kHz)', fontsize=12)
    ax_bot.set_ylabel('ΔPSD (dB)', fontsize=12)
    ax_bot.legend(fontsize=8)
    ax_bot.grid(True, alpha=0.2)
    ax_bot.set_xlim(0, 16)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/psd/dpsd_ref_all_sp{sp}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  dpsd_ref_all_sp{sp}.png")

    # Per-sample data for CSV
    for sname in selected:
        wav_name = sname + '.wav'
        if wav_name in mel_cache:
            mel_db = mel_cache[wav_name]
            inp_psd = mel_to_psd(mel_db)
            for (reg_type, lam) in configs:
                p = get_npy_path(reg_type, lam, sp, sname)
                if p and os.path.exists(p):
                    rec_psd = mel_to_psd(np.load(p))
                    for fb, freq_hz in enumerate(freqs):
                        all_rows.append({
                            'mode': mode, 'split_point': sp,
                            'regularizer': reg_type, 'lambda': lam,
                            'sample': sname, 'freq_hz': freq_hz,
                            'psd_input': inp_psd[fb],
                            'psd_recon': rec_psd[fb],
                            'psd_diff': rec_psd[fb] - inp_psd[fb],
                        })

if all_rows:
    df_psd = pd.DataFrame(all_rows)
    df_psd.to_csv(f'{OUTPUT_DIR}/psd/psd_data.csv', index=False)
    print(f"  Saved {OUTPUT_DIR}/psd/psd_data.csv ({len(df_psd)} rows)")

    # Band summary
    band_labels = [(0, 1000, '0-1kHz'), (1000, 4000, '1-4kHz'),
                   (4000, 8000, '4-8kHz'), (8000, 15000, '8-15kHz'),
                   (15000, 16000, '15-16kHz')]
    band_rows = []
    for (m, sp, reg_type, lam), grp in df_psd.groupby(['mode', 'split_point', 'regularizer', 'lambda']):
        for lo, hi, blabel in band_labels:
            sub = grp[(grp['freq_hz'] >= lo) & (grp['freq_hz'] < hi)]
            if len(sub) == 0:
                continue
            band_rows.append({
                'mode': m, 'split_point': sp,
                'regularizer': reg_type, 'lambda': lam,
                'band': blabel, 'freq_lo': lo, 'freq_hi': hi,
                'dpsd_mean': sub['psd_diff'].mean(),
                'dpsd_std': sub['psd_diff'].std(),
                'psd_input_mean': sub['psd_input'].mean(),
                'psd_recon_mean': sub['psd_recon'].mean(),
            })
    pd.DataFrame(band_rows).to_csv(f'{OUTPUT_DIR}/psd/psd_band_summary.csv', index=False)
    print(f"  Saved {OUTPUT_DIR}/psd/psd_band_summary.csv")

print("\nAll comparison done!")
