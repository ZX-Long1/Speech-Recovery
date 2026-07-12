import os, sys, glob, random
import numpy as np
import torch
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sc_evaluation_nTV import feat_mel, SAMPLE_RATE, N_MELS, \
    MEL_MIN_VAL, MEL_MAX_VAL, MAX_COLUMN, TEST_WAV, N_FFT

NTV_LIST = ['0.0', '0.5', '1.0', '1.5', '2.0']
N_SAMPLES = 5
FIGS_DIR = 'sc_eval_figures'
os.makedirs(FIGS_DIR, exist_ok=True)

random.seed(42)

def load_original_mel(wav_name):
    wav_path = os.path.join(TEST_WAV, wav_name)
    audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)
    return feat_mel(audio)

for ntv in NTV_LIST:
    base = f'sc_eval_results_mel_nTV{ntv}/reconstructed_mel'
    for mode in ['whitebox_shallow', 'blackbox_shallow']:
        for sp in [1, 2, 3, 4]:
            pattern = f'{base}/{mode}/sp{sp}/*.npy'
            npy_files = sorted(glob.glob(pattern))
            if not npy_files:
                continue
            selected = random.sample(npy_files, min(N_SAMPLES, len(npy_files)))
            for idx, npy_path in enumerate(selected, 1):
                wav_name = os.path.basename(npy_path).replace('.npy', '.wav')
                rec_db = np.load(npy_path)  # (64, 96), dB scale
                orig_db = load_original_mel(wav_name)

                fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                for ax, data, ti in zip(axes,
                    [orig_db, rec_db],
                    ['Original', 'Reconstructed']):
                    img = librosa.display.specshow(
                        data, sr=SAMPLE_RATE, hop_length=N_FFT // 4,
                        x_axis='time', y_axis='mel', ax=ax,
                        cmap='magma')
                    ax.set_title(f'{ti} (nTV={ntv}, {mode}, sp{sp})')
                    ax.set_ylabel('Mel Frequency')
                    ax.set_xlabel('Time (s)')
                out_name = f'nTV{ntv}_{mode}_sp{sp}_{idx:02d}.png'
                out_path = os.path.join(FIGS_DIR, out_name)
                plt.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'Saved {out_path}', flush=True)

print(f'\nDone! All figures in {FIGS_DIR}/')
