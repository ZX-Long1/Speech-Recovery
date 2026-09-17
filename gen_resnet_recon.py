import os, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cremad_experiment.cremad_models import ResNet18, BasicBlock
from cremad_experiment.cremad_models import Conv_sp1, Conv_sp2, Conv_sp3, Conv_sp4
from cremad_experiment.cremad_models import Conv_sp1_shallow, Conv_sp2_shallow, Conv_sp3_shallow, Conv_sp4_shallow

parser = argparse.ArgumentParser()
parser.add_argument('--stems_file', type=str,
                    default='/media/sda1/zxlong/tmc_re/results/speaker_leakage/query_stems_88.txt')
parser.add_argument('--modes', nargs='+', default=['whitebox_shallow', 'blackbox'])
parser.add_argument('--device_id', type=int, default=0)
parser.add_argument('--ckpt_root', type=str,
                    default='/media/sda1/zxlong/tmc_re/CREMA-D/inversion/mel',
                    help='Root dir containing {mode}/split_point{sp}/train_record/inversion.pth')
parser.add_argument('--out_base', type=str,
                    default='/media/sda1/zxlong/tmc_re/CREMA-D/eval_results/audio_samples',
                    help='Base dir for generated audio')
parser.add_argument('--mode_label', type=str, default='',
                    help='If set, output to {out_base}/{mode_label} instead of mel_{mode}')
args = parser.parse_args()

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
WAV_DIR = f'{BASE}/AudioWAV'
SAMPLE_RATE = 16000
N_FFT = 320
HOP_LENGTH = 80
N_MELS = 64
MAX_TIME = 640
DEVICE = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

norm_info = json.load(open(f'{BASE}/processed/norm_info.json'))
MEL_MIN, MEL_MAX = norm_info['mel_min'], norm_info['mel_max']

INV_CLS = {1: Conv_sp1, 2: Conv_sp2, 3: Conv_sp3, 4: Conv_sp4}
INV_CLS_SHALLOW = {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow, 4: Conv_sp4_shallow}
SPLITS = {'whitebox_shallow': [1, 2, 3, 4], 'blackbox': [1, 2, 3]}
CKPT_DIR = {m: f'{args.ckpt_root}/{m}' for m in SPLITS}
OUT_BASE = args.out_base

with open(args.stems_file) as f:
    stems = [l.strip() for l in f if l.strip()]
print(f'Stems: {len(stems)}  Device: {DEVICE}', flush=True)

def feat_mel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_fft=N_FFT,
                                          n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    if mel_db.shape[1] > MAX_TIME:
        return mel_db[:, :MAX_TIME]
    import scipy.interpolate as sci
    h, w = mel_db.shape
    padded = np.zeros((h, MAX_TIME))
    padded[:, :w] = mel_db
    for i in range(h):
        x_new = np.linspace(0, w - 1, MAX_TIME)
        x_old = np.arange(w)
        fi = sci.interp1d(x_old, mel_db[i], kind='linear', fill_value='extrapolate')
        padded[i] = fi(x_new)
    return padded

def normalize(feat, mn, mx):
    norm = ((feat - mn) / (mx - mn)) * 255
    return torch.from_numpy(np.uint8(norm)).unsqueeze(0).float() / 255.0

def denormalize(tensor, mn, mx):
    arr = tensor.squeeze().cpu().numpy()
    return arr * (mx - mn) + mn

def reconstruct_audio_from_mel(mel_db_np):
    mel_power = librosa.db_to_power(mel_db_np)
    return librosa.feature.inverse.mel_to_audio(
        mel_power, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        power=2.0, n_iter=32)

classifier = ResNet18(BasicBlock, 1, 6).to(DEVICE)
cl_path = f'{BASE}/classifier/mel_train_record/classifier.pth'
ckpt = torch.load(cl_path, map_location='cpu')
classifier.load_state_dict(ckpt['model'])
classifier.eval()
print(f'Loaded classifier (epoch={ckpt.get("epoch", "?")})', flush=True)

for mode in args.modes:
    out_dir = f'{OUT_BASE}/{args.mode_label}' if args.mode_label else f'{OUT_BASE}/mel_{mode}'
    os.makedirs(out_dir, exist_ok=True)
    for sp in SPLITS[mode]:
        inv_cls = INV_CLS_SHALLOW
        inversion = inv_cls[sp]().to(DEVICE)
        ckpt_path = f'{CKPT_DIR[mode]}/split_point{sp}/train_record/inversion.pth'
        if not os.path.exists(ckpt_path):
            print(f'  [SKIP] {mode} sp{sp}: no checkpoint at {ckpt_path}', flush=True)
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu')
        inversion.load_state_dict(ckpt['model'])
        inversion.eval()
        print(f'Generating {mode} sp{sp} ...', flush=True)
        for stem in stems:
            recon_path = f'{out_dir}/recon_sp{sp}_{stem}.wav'
            clean_path = f'{out_dir}/clean_sp1_{stem}.wav'
            wav_path = f'{WAV_DIR}/{stem}.wav'
            if not os.path.exists(wav_path):
                print(f'  missing source wav: {stem}', flush=True)
                continue
            audio, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            if not os.path.exists(clean_path):
                sf.write(clean_path, audio, SAMPLE_RATE)
            if os.path.exists(recon_path):
                continue
            feat_db = feat_mel(audio)
            inp = normalize(feat_db, MEL_MIN, MEL_MAX).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                ft = classifier(inp, split_point=sp)
                rec = inversion(ft)
            recon_db = denormalize(rec, MEL_MIN, MEL_MAX)
            recon_audio = reconstruct_audio_from_mel(recon_db)
            sf.write(recon_path, recon_audio, SAMPLE_RATE)
        print(f'  done {mode} sp{sp}', flush=True)

print('All done.')
