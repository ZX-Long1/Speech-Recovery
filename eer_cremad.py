import os, sys, argparse, warnings
from collections import Counter
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from sklearn.metrics import roc_curve

sys.path.insert(0, '/media/sda1/zxlong/tmc_re')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import build_classifier, build_decoder, CREMAD, REPRO

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['resnet18', 'whisper'], default='resnet18')
parser.add_argument('--mode', choices=['whitebox', 'blackbox'], default='whitebox')
parser.add_argument('--split_point', type=int, choices=[1, 3], default=1)
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--stems_file', type=str,
                    default='/media/sda1/zxlong/tmc_re/results/speaker_leakage/query_stems_88.txt')
parser.add_argument('--subset', type=int, default=0, help='0 = all stems')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

device = torch.device(f'cuda:{args.gpu}')
torch.cuda.set_device(args.gpu)

import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
os.environ['HF_HUB_OFFLINE'] = '1'
spk_enc = EncoderClassifier.from_hparams(
    source='/media/sda1/zxlong/tmc_re/model/spkrec-ecapa-voxceleb',
    savedir='/media/sda1/zxlong/tmc_re/model/spkrec-ecapa-voxceleb',
    hparams_file='hyperparams_local.yaml',
    run_opts={'device': str(device)})

stems = [l.strip() for l in open(args.stems_file) if l.strip()]
if args.subset > 0:
    stems = stems[:args.subset]

if args.model == 'resnet18':
    classifier = build_classifier('cremad', device)
    decoder = build_decoder(args.split_point, device)
    ck = torch.load(args.ckpt, map_location='cpu')
    decoder.load_state_dict(ck['model'])
else:
    import whisper
    from cremad_experiment.whisper_models import (WhisperEncoder,
                                                  Decoder_whitebox,
                                                  Decoder_blackbox)
    encoder = WhisperEncoder(split_point=args.split_point, freeze=True).to(device)
    encoder.eval()
    decoder = (Decoder_whitebox if args.mode == 'whitebox'
               else Decoder_blackbox)().to(device)
    ck = torch.load(args.ckpt, map_location='cpu')
    decoder.load_state_dict(ck['model'])
decoder.eval()

out_dir = f'{REPRO}/eer_audio/{args.model}_{args.mode}_sp{args.split_point}'
os.makedirs(out_dir, exist_ok=True)

norm_info = json_ = None
import json as _json
norm_info = _json.load(open(f'{CREMAD}/processed/norm_info.json'))


def recon_mel_to_audio_db(recon_np):
    power = librosa.db_to_power(recon_np)
    return librosa.feature.inverse.mel_to_audio(power, sr=16000, n_fft=320,
                                                hop_length=80, power=2.0, n_iter=32)


def recon_logmel_to_audio(recon_np):
    power = np.power(10.0, recon_np)
    S = librosa.feature.inverse.mel_to_stft(power, sr=16000, n_fft=400, power=2.0)
    return librosa.griffinlim(S, hop_length=160, n_iter=32)


for stem in stems:
    recon_path = f'{out_dir}/recon_sp{args.split_point}_{stem}.wav'
    if os.path.exists(recon_path):
        continue
    wav_path = f'{CREMAD}/AudioWAV/{stem}.wav'
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    sf.write(f'{out_dir}/clean_sp1_{stem}.wav', audio, 16000)
    if args.model == 'resnet18':
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_fft=320,
                                             n_mels=64, hop_length=80)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] > 640:
            mel_db = mel_db[:, :640]
        else:
            import scipy.interpolate as sci
            h, w = mel_db.shape
            padded = np.zeros((h, 640))
            padded[:, :w] = mel_db
            for i in range(h):
                fi = sci.interp1d(np.arange(w), mel_db[i], kind='linear',
                                  fill_value='extrapolate')
                padded[i] = fi(np.linspace(0, w - 1, 640))
            mel_db = padded
        mn, mx = norm_info['mel_min'], norm_info['mel_max']
        inp = torch.from_numpy(np.uint8(((mel_db - mn) / (mx - mn)) * 255)) \
                .unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
        with torch.no_grad():
            pred = classifier(inp, split_point=args.split_point)
            recon = decoder(pred)
        recon_db = recon.squeeze().cpu().numpy() * (mx - mn) + mn
        r_audio = recon_mel_to_audio_db(recon_db)
    else:
        mel = whisper.log_mel_spectrogram(audio, padding=0).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = encoder(mel)
            recon = decoder(feat)
        t = min(recon.shape[-1], mel.shape[-1])
        r_audio = recon_logmel_to_audio(recon[:, :, :t].squeeze(0).cpu().numpy())
    sf.write(recon_path, r_audio, 16000)
print(f'generated recon wavs: {len(stems)} -> {out_dir}', flush=True)


def load_audio(path):
    a, sr = sf.read(path)
    if a.ndim > 1:
        a = np.mean(a, axis=0)
    if sr != 16000:
        import scipy.signal
        a = scipy.signal.resample(a, int(len(a) * 16000 / sr))
    return a / (np.max(np.abs(a)) + 1e-10)


def extract(a):
    with torch.no_grad():
        w = torch.from_numpy(a).float().to(device).unsqueeze(0)
        emb = spk_enc.encode_batch(w, wav_lens=torch.tensor([1.0]).to(device))
        return emb.squeeze(0).squeeze(0).cpu().numpy()


enroll_emb = {}
enroll_sum = {}
enroll_count = {}
for fn in sorted(os.listdir(f'{CREMAD}/AudioWAV')):
    if not fn.endswith('.wav'):
        continue
    stem = fn[:-4]
    sp = stem.split('_')[0]
    e = extract(load_audio(f'{CREMAD}/AudioWAV/{fn}'))
    enroll_emb.setdefault(sp, {})[stem] = e
    enroll_sum.setdefault(sp, np.zeros_like(e))
    enroll_sum[sp] += e
    enroll_count[sp] = enroll_count.get(sp, 0) + 1
full_proto = {s: enroll_sum[s] / enroll_count[s] for s in enroll_sum}


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def target_proto(spk, excl):
    if spk not in enroll_emb or excl not in enroll_emb[spk]:
        return None
    if enroll_count[spk] <= 1:
        return None
    return (enroll_sum[spk] - enroll_emb[spk][excl]) / (enroll_count[spk] - 1)


scores, labels = [], []
skipped = 0
for stem in stems:
    q_path = f'{out_dir}/recon_sp{args.split_point}_{stem}.wav'
    if not os.path.exists(q_path):
        skipped += 1
        continue
    spk = stem.split('_')[0]
    proto = target_proto(spk, stem)
    if proto is None:
        skipped += 1
        continue
    q = extract(load_audio(q_path))
    scores.append(cos(q, proto))
    labels.append(1)
    for o in full_proto:
        if o == spk:
            continue
        scores.append(cos(q, full_proto[o]))
        labels.append(0)

fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
fnr = 1 - tpr
idx = np.nanargmin(np.abs(fpr - fnr))
eer = (fpr[idx] + fnr[idx]) / 2 * 100
print(f'EER (strict LOO) = {eer:.2f}%  (queries={len(stems)-skipped}, '
      f'trials={len(scores)}, skipped={skipped})')
