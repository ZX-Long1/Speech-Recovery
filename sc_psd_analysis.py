import os, sys, gc
import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from glob import glob
import warnings
warnings.filterwarnings('ignore')

SC_BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'
TEST_WAV = f'{SC_BASE}/audio/test_set'
SAMPLE_RATE = 49050
N_MELS = 64
N_FFT = int(0.02 * SAMPLE_RATE)
HOP_LENGTH = N_FFT // 4
MAX_COLUMN = 96
MEL_MIN_VAL = -80.00000381469727
MEL_MAX_VAL = 3.814697265625e-06

OUTPUT_DIR = 'sc_psd_results'
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)

NTV_VALS = [0.0, 0.5, 1.0, 1.5, 2.0]
MODES = ['whitebox_shallow', 'blackbox_shallow']
SPLIT_POINTS = [1, 2, 3, 4]
N_SAMPLES = 30

COLORS = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']  # magma-like

def feat_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] > MAX_COLUMN:
        return mel_db[:, :MAX_COLUMN]
    from scipy.interpolate import interp1d
    h, w = mel_db.shape
    padded = np.zeros((h, MAX_COLUMN))
    padded[:, :w] = mel_db
    for i in range(h):
        x_new = np.linspace(0, w - 1, MAX_COLUMN)
        x_old = np.arange(w)
        fi = interp1d(x_old, mel_db[i], kind='linear', fill_value='extrapolate')
        padded[i] = fi(x_new)
    return padded

def compute_input_mel(wav_name):
    wav_path = os.path.join(TEST_WAV, wav_name)
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    return feat_mel(y)

def mel_to_psd(mel_db, M_pinv):
    power_mel = 10.0 ** (mel_db / 10.0)
    linear_power = M_pinv @ power_mel
    linear_power = np.maximum(linear_power, 1e-12)
    psd = np.mean(linear_power, axis=1)
    return 10.0 * np.log10(psd)

print("Pre-computing mel filterbank pseudo-inverse...")
M = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS)
M_pinv = np.linalg.pinv(M)
freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)

all_results = []
os.makedirs(f'{OUTPUT_DIR}/figures', exist_ok=True)

for mode in MODES:
    for sp in SPLIT_POINTS:
        print(f"\n{'='*60}")
        print(f"Processing: mode={mode}, sp={sp}")
        
        # Find common samples across all nTV for this (mode, sp)
        sample_sets = []
        for ntv in NTV_VALS:
            d = f'sc_eval_results_mel_nTV{ntv}/reconstructed_mel/{mode}/sp{sp}'
            if not os.path.isdir(d):
                sample_sets.append(set())
                continue
            files = [f.replace('.npy', '') for f in os.listdir(d) if f.endswith('.npy')]
            sample_sets.append(set(files))
        
        common = set.intersection(*sample_sets) if all(s for s in sample_sets) else set()
        if len(common) == 0:
            print(f"  WARNING: no common samples for {mode} sp{sp}")
            continue
        
        selected = sorted(common)[:N_SAMPLES]
        print(f"  Common samples: {len(common)}, using: {len(selected)}")
        
        # Compute input PSDs (shared across all nTV)
        input_psds = {}
        for sname in selected:
            wav = sname + '.wav'
            mel_db = compute_input_mel(wav)
            psd = mel_to_psd(mel_db, M_pinv)
            input_psds[sname] = psd
        
        input_mean = np.mean(list(input_psds.values()), axis=0)
        input_std = np.std(list(input_psds.values()), axis=0)
        
        # Compute PSD per nTV
        ntv_results = {}
        for ntv in NTV_VALS:
            d = f'sc_eval_results_mel_nTV{ntv}/reconstructed_mel/{mode}/sp{sp}'
            psds = []
            for sname in selected:
                fpath = os.path.join(d, f'{sname}.npy')
                if not os.path.isfile(fpath):
                    continue
                recon_mel = np.load(fpath)
                psd = mel_to_psd(recon_mel, M_pinv)
                psds.append(psd)
            if len(psds) == 0:
                continue
            ntv_results[ntv] = {
                'mean': np.mean(psds, axis=0),
                'std': np.std(psds, axis=0),
                'individual': np.array(psds)
            }
            print(f"    nTV={ntv}: {len(psds)} samples")
        
        # Save per-sample data
        for i, sname in enumerate(selected):
            for ntv in NTV_VALS:
                d = f'sc_eval_results_mel_nTV{ntv}/reconstructed_mel/{mode}/sp{sp}'
                fpath = os.path.join(d, f'{sname}.npy')
                if not os.path.isfile(fpath):
                    continue
                recon_mel = np.load(fpath)
                psd_rec = mel_to_psd(recon_mel, M_pinv)
                psd_inp = input_psds.get(sname, np.full(len(freqs), np.nan))
                for fb in range(len(freqs)):
                    all_results.append({
                        'mode': mode, 'split_point': sp, 'nTV': ntv,
                        'sample': sname, 'freq_bin': fb,
                        'freq_hz': freqs[fb],
                        'psd_input': psd_inp[fb],
                        'psd_recon': psd_rec[fb],
                        'psd_diff': psd_rec[fb] - psd_inp[fb]
                    })
        
        # === Figures ===
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Input PSD
        ax.plot(freqs / 1000, input_mean, color='k', linewidth=2.5, label='Input', alpha=0.7)
        ax.fill_between(freqs / 1000, input_mean - input_std, input_mean + input_std,
                         color='k', alpha=0.08)
        
        # nTV PSDs
        for ntv, color in zip(NTV_VALS, COLORS):
            if ntv not in ntv_results:
                continue
            r = ntv_results[ntv]
            ax.plot(freqs / 1000, r['mean'], color=color, linewidth=1.8, label=f'nTV={ntv}')
            ax.fill_between(freqs / 1000, r['mean'] - r['std'], r['mean'] + r['std'],
                            color=color, alpha=0.06)
        
        ax.set_xlabel('Frequency (kHz)', fontsize=12)
        ax.set_ylabel('Power Spectral Density (dB)', fontsize=12)
        ax.set_title(f'PSD Comparison — {mode}, Split Point {sp}', fontsize=13)
        ax.legend(fontsize=10, loc='lower left')
        ax.set_xlim(0, freqs[-1] / 1000)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/figures/psd_{mode}_sp{sp}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # ΔPSD figure
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        for ntv, color in zip(NTV_VALS, COLORS):
            if ntv not in ntv_results:
                continue
            r = ntv_results[ntv]
            diff_mean = r['mean'] - input_mean
            ax2.plot(freqs / 1000, diff_mean, color=color, linewidth=1.8, label=f'nTV={ntv}')
        
        ax2.axhline(y=0, color='k', linewidth=0.8, linestyle='--')
        ax2.set_xlabel('Frequency (kHz)', fontsize=12)
        ax2.set_ylabel('ΔPSD (dB)', fontsize=12)
        ax2.set_title(f'PSD Difference (Recon − Input) — {mode}, Split Point {sp}', fontsize=13)
        ax2.legend(fontsize=10)
        ax2.set_xlim(0, freqs[-1] / 1000)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(f'{OUTPUT_DIR}/figures/dpsd_{mode}_sp{sp}.png', dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        # Frequency band summary
        bands = [(0, 1000, '0-1kHz'), (1000, 4000, '1-4kHz'),
                 (4000, 8000, '4-8kHz'), (8000, 15000, '8-15kHz'),
                 (15000, 24500, '15-24.5kHz')]
        band_data = []
        for ntv, color in zip(NTV_VALS, COLORS):
            if ntv not in ntv_results:
                continue
            r = ntv_results[ntv]
            diff_mean = r['mean'] - input_mean
            for lo, hi, label in bands:
                mask = (freqs >= lo) & (freqs < hi)
                if mask.sum() > 0:
                    band_data.append({
                        'mode': mode, 'split_point': sp, 'nTV': ntv,
                        'band': label, 'freq_lo': lo, 'freq_hi': hi,
                        'dpsd_mean': diff_mean[mask].mean(),
                        'dpsd_std': r['std'][mask].mean()})
        pd.DataFrame(band_data).to_csv(
            f'{OUTPUT_DIR}/figures/band_{mode}_sp{sp}.csv', index=False)
        
        # Heatmap: nTV × freq_bin
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ntv_list = [n for n in NTV_VALS if n in ntv_results]
        if ntv_list:
            heat_data = np.array([ntv_results[n]['mean'] - input_mean for n in ntv_list])
            im = ax3.imshow(heat_data, aspect='auto', cmap='RdBu_r',
                            extent=[0, freqs[-1]/1000, len(ntv_list)-0.5, -0.5],
                            interpolation='bilinear')
            ax3.set_yticks(range(len(ntv_list)))
            ax3.set_yticklabels([f'nTV={n}' for n in ntv_list])
            ax3.set_xlabel('Frequency (kHz)', fontsize=12)
            ax3.set_ylabel('nTV', fontsize=12)
            ax3.set_title(f'ΔPSD (dB) — {mode}, Split Point {sp}', fontsize=13)
            fig3.colorbar(im, ax=ax3, shrink=0.8)
        fig3.tight_layout()
        fig3.savefig(f'{OUTPUT_DIR}/figures/heatmap_{mode}_sp{sp}.png', dpi=150, bbox_inches='tight')
        plt.close(fig3)
        
        gc.collect()

# Save all data
df = pd.DataFrame(all_results)
df.to_csv(f'{OUTPUT_DIR}/psd_data.csv', index=False)
print(f"\n\nSaved: {OUTPUT_DIR}/psd_data.csv ({len(df)} rows)")

# Summary per (mode, sp, nTV, freq_bin)
summary = df.groupby(['mode', 'split_point', 'nTV', 'freq_bin', 'freq_hz'])[
    ['psd_input', 'psd_recon', 'psd_diff']].mean().reset_index()
summary.to_csv(f'{OUTPUT_DIR}/psd_summary.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/psd_summary.csv")

# Aggregate band summary
band_labels = [(0, 1000, '0-1kHz'), (1000, 4000, '1-4kHz'),
               (4000, 8000, '4-8kHz'), (8000, 15000, '8-15kHz'),
               (15000, 24500, '15-24.5kHz')]
band_rows = []
for (mode, sp, ntv), grp in df.groupby(['mode', 'split_point', 'nTV']):
    for lo, hi, blabel in band_labels:
        sub = grp[(grp['freq_hz'] >= lo) & (grp['freq_hz'] < hi)]
        if len(sub) == 0: continue
        band_rows.append({
            'mode': mode, 'split_point': sp, 'nTV': ntv,
            'band': blabel, 'freq_lo': lo, 'freq_hi': hi,
            'dpsd_mean': sub['psd_diff'].mean(),
            'dpsd_std': sub['psd_diff'].std(),
            'psd_input_mean': sub['psd_input'].mean(),
            'psd_recon_mean': sub['psd_recon'].mean()})
pd.DataFrame(band_rows).to_csv(f'{OUTPUT_DIR}/psd_band_summary.csv', index=False)
print(f"Saved: {OUTPUT_DIR}/psd_band_summary.csv")

# Create grid overview figure
combos = [(m, s) for m in MODES for s in SPLIT_POINTS]
n_combos = len(combos)
n_cols = 4
n_rows = (n_combos + n_cols - 1) // n_cols
fig_grid, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
axes = axes.flatten()
for idx, (mode, sp) in enumerate(combos):
    sub = df[(df['mode'] == mode) & (df['split_point'] == sp) &
             (df['sample'] == df['sample'].unique()[0])]
    if len(sub) == 0: continue
    for ntv, color in zip(NTV_VALS, COLORS):
        s = df[(df['mode'] == mode) & (df['split_point'] == sp) & (df['nTV'] == ntv)]
        if len(s) == 0: continue
        mean_psd = s.groupby('freq_hz')['psd_recon'].mean()
        ax.plot(mean_psd.index / 1000, mean_psd.values, color=color, lw=1.2, label=f'nTV={ntv}')
    inp = df[(df['mode'] == mode) & (df['split_point'] == sp)]
    inp_mean = inp.groupby('freq_hz')['psd_input'].mean()
    ax.plot(inp_mean.index / 1000, inp_mean.values, 'k', lw=2, label='Input', alpha=0.6)
    ax.set_title(f'{mode}, sp={sp}', fontsize=11)
    ax.set_xlim(0, 24.5)
    ax.grid(True, alpha=0.2)
    if idx == 0: ax.legend(fontsize=8, loc='lower left')
for idx in range(n_combos, len(axes)):
    axes[idx].set_visible(False)
fig_grid.tight_layout()
fig_grid.savefig(f'{OUTPUT_DIR}/figures/psd_grid.png', dpi=150, bbox_inches='tight')
plt.close(fig_grid)

# === Dual-panel PSD + ΔPSD reference figures ===
print("\nGenerating dpsd_ref figures (dual-panel with input PSD)...")
for mode in MODES:
    for sp in SPLIT_POINTS:
        sub = df[(df['mode'] == mode) & (df['split_point'] == sp)]
        if len(sub) == 0:
            continue
        
        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        
        # Aggregate PSDs
        inp_mean = sub.groupby('freq_hz')['psd_input'].mean()
        ntv_means = {}
        for ntv in NTV_VALS:
            s = sub[sub['nTV'] == ntv]
            if len(s) == 0:
                continue
            ntv_means[ntv] = s.groupby('freq_hz')['psd_recon'].mean()
        
        # Upper: PSD curves
        ax_top.plot(freqs / 1000, inp_mean.values, color='k', linewidth=2.5,
                    label='Input', alpha=0.7, zorder=5)
        for ntv, color in zip(NTV_VALS, COLORS):
            if ntv not in ntv_means:
                continue
            ax_top.plot(freqs / 1000, ntv_means[ntv].values, color=color,
                        linewidth=1.8, label=f'nTV={ntv}')
        ax_top.set_ylabel('PSD (dB)', fontsize=12)
        ax_top.set_title(f'PSD + ΔPSD — {mode}, Split Point {sp}', fontsize=13)
        ax_top.legend(fontsize=9, loc='lower left')
        ax_top.grid(True, alpha=0.2)
        ax_top.set_xlim(0, freqs[-1] / 1000)
        
        # Lower: ΔPSD curves
        for ntv, color in zip(NTV_VALS, COLORS):
            if ntv not in ntv_means:
                continue
            delta = ntv_means[ntv].values - inp_mean.values
            ax_bot.plot(freqs / 1000, delta, color=color, linewidth=1.8, label=f'nTV={ntv}')
        
        ax_bot.axhline(y=0, color='k', linewidth=0.8, linestyle='--', alpha=0.5)
        for thresh in [-3, 3]:
            ax_bot.axhline(y=thresh, color='gray', linewidth=0.5, linestyle=':', alpha=0.4)
        ax_bot.set_xlabel('Frequency (kHz)', fontsize=12)
        ax_bot.set_ylabel('ΔPSD (dB)', fontsize=12)
        ax_bot.legend(fontsize=9)
        ax_bot.grid(True, alpha=0.2)
        ax_bot.set_xlim(0, freqs[-1] / 1000)
        
        fig.tight_layout()
        fig.savefig(f'{OUTPUT_DIR}/figures/dpsd_ref_{mode}_sp{sp}.png',
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  dpsd_ref_{mode}_sp{sp}.png")

print("\nAll done!")
