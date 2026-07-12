import os, sys, random
import numpy as np
import librosa
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
import json

SRC_ROOT = os.path.expanduser('/media/sda1/zxlong/tmc_re/CREMA-D/AudioWAV')
OUT_ROOT = '/media/sda1/zxlong/tmc_re/CREMA-D/processed'

SAMPLE_RATE = 16000
N_MELS = 64
N_MFCC = 64
N_FFT = 320
HOP_LENGTH = 80
MAX_TIME = 640

EMO_MAP = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}

def extract_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
                                          n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return _pad_or_trim(mel_db)

def extract_mfcc(y):
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
        x_new = np.linspace(0, w - 1, MAX_TIME)
        x_old = np.arange(w)
        fi = sci.interp1d(x_old, feat[i, :], kind='linear', fill_value='extrapolate')
        padded[i, :] = fi(x_new)
    return padded

def main():
    os.makedirs(f'{OUT_ROOT}/train', exist_ok=True)
    os.makedirs(f'{OUT_ROOT}/test', exist_ok=True)
    os.makedirs(f'{OUT_ROOT}/aux', exist_ok=True)

    files = sorted(glob(f'{SRC_ROOT}/*.wav'))
    print(f'Total files: {len(files)}')

    by_emotion = defaultdict(list)
    for fp in files:
        basename = os.path.basename(fp)
        emo = basename.split('_')[2]
        label = EMO_MAP[emo]
        by_emotion[label].append(fp)

    train_files, test_files, aux_files = [], [], []
    for label, flist in sorted(by_emotion.items()):
        tr, rest = train_test_split(flist, test_size=0.3, random_state=42)
        te, aux = train_test_split(rest, test_size=0.5, random_state=42)
        train_files.extend(tr)
        test_files.extend(te)
        aux_files.extend(aux)
        print(f'  {label}: {len(tr)}/{len(te)}/{len(aux)}')

    mel_min, mel_max = float('inf'), float('-inf')
    mfcc_min, mfcc_max = float('inf'), float('-inf')

    for split_name, flist in [('train', train_files), ('test', test_files), ('aux', aux_files)]:
        for fp in flist:
            basename = os.path.basename(fp)
            label = EMO_MAP[basename.split('_')[2]]

            y, _ = librosa.load(fp, sr=SAMPLE_RATE, mono=True)

            mel_db = extract_mel(y)
            mfcc = extract_mfcc(y)

            mel_min = min(mel_min, mel_db.min())
            mel_max = max(mel_max, mel_db.max())
            mfcc_min = min(mfcc_min, mfcc.min())
            mfcc_max = max(mfcc_max, mfcc.max())

            stem = os.path.splitext(basename)[0]
            np.save(f'{OUT_ROOT}/{split_name}/{stem}_mel.npy', mel_db)
            np.save(f'{OUT_ROOT}/{split_name}/{stem}_mfcc.npy', mfcc)

    norm_info = {
        'mel_min': float(mel_min), 'mel_max': float(mel_max),
        'mfcc_min': float(mfcc_min), 'mfcc_max': float(mfcc_max),
    }
    json.dump(norm_info, open(f'{OUT_ROOT}/norm_info.json', 'w'), indent=2)
    print(f'\n  mel: [{mel_min:.4f}, {mel_max:.4f}]')
    print(f'  mfcc: [{mfcc_min:.4f}, {mfcc_max:.4f}]')
    print(f'\n  train={len(train_files)}, test={len(test_files)}, aux={len(aux_files)}')
    print('Done.')

if __name__ == '__main__':
    main()
