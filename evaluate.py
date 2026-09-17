import os, sys, re, argparse, json
import numpy as np
import torch
import torch.nn.functional as F
import librosa

sys.path.insert(0, '/media/sda1/zxlong/tmc_re')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import (build_classifier, build_decoder, build_loaders,
                  per_sample_snr, per_pixel_mse, CREMAD, REPRO)

SENTENCE_MAP = {
    'IEO': "it's eleven o'clock", 'TIE': 'that is exactly what happened',
    'IOM': "i'm on my way to the meeting", 'IWW': 'i wonder what this is about',
    'TAI': 'the airplane is almost full', 'MTI': 'maybe tomorrow it will be cold',
    'IWL': 'i would like a new alarm clock', 'ITH': "i think i have a doctor's appointment",
    'DFA': "don't forget a jacket", 'ITS': "i think i've seen this before",
    'TSI': 'the surface is slick', 'WSI': "we'll stop in a couple of minutes",
}

def norm(t):
    t = t.lower().replace("'", "")
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    return ' '.join(t.split())

def edist(r, h):
    r, h = r.split(), h.split()
    d = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        d[i][0] = i
    for j in range(len(h) + 1):
        d[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    return d[len(r)][len(h)]


def recon_mel_to_audio(recon_np, sr=16000, n_fft=320, hop=80):
    power = librosa.db_to_power(recon_np)
    return librosa.feature.inverse.mel_to_audio(power, sr=sr, n_fft=n_fft,
                                                hop_length=hop, power=2.0, n_iter=32)


def recon_logmel_to_audio(recon_np):
    power = np.power(10.0, recon_np)
    S = librosa.feature.inverse.mel_to_stft(power, sr=16000, n_fft=400, power=2.0)
    return librosa.griffinlim(S, hop_length=160, n_iter=32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['sc', 'cremad'], default='cremad')
    p.add_argument('--model', choices=['resnet18', 'whisper'], default='resnet18')
    p.add_argument('--mode', choices=['whitebox', 'blackbox'], default='whitebox')
    p.add_argument('--split_point', type=int, choices=[1, 3], default=1)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--wer', action='store_true')
    p.add_argument('--wer_subset', type=int, default=20)
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()

    device = torch.device(f'cuda:{args.gpu}')
    torch.cuda.set_device(args.gpu)
    whisper_model = None
    if args.wer:
        import whisper
        whisper_model = whisper.load_model('small', device=device)
    _, test_set, _ = build_loaders(args.dataset, smoke=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                              shuffle=False, num_workers=4)

    if args.model == 'resnet18':
        classifier = build_classifier(args.dataset, device)
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
        if whisper_model is None:
            whisper_model = whisper.load_model('small', device=device)

    decoder.eval()
    mse_sum = snr_sum = 0.0
    n = 0
    wer_items = []
    if args.model == 'whisper':
        test_stems = [l.strip() for l in open(f'{REPRO}/manifests/cremad_test.txt')]
        for idx, rel in enumerate(test_stems):
            stem = os.path.basename(rel).replace('_mel.npy', '')
            audio, _ = librosa.load(f'{CREMAD}/AudioWAV/{stem}.wav', sr=16000, mono=True)
            mel = whisper.log_mel_spectrogram(audio, padding=0).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = encoder(mel)
                recon = decoder(feat)
            t = min(recon.shape[-1], mel.shape[-1])
            recon = recon[:, :, :t]
            mel_t = mel[:, :, :t]
            mse_sum += F.mse_loss(recon, mel_t).item()
            snr_sum += per_sample_snr(mel_t, recon)
            n += 1
            if args.wer and len(wer_items) < args.wer_subset:
                r_audio = recon_logmel_to_audio(recon.squeeze(0).cpu().numpy())
                hyp = whisper_model.transcribe(r_audio, language='en',
                                               task='transcribe',
                                               fp16=True)['text'].strip()
                gt = SENTENCE_MAP[stem.split('_')[1]]
                e = edist(norm(gt), norm(hyp))
                wer_items.append((e, len(norm(gt).split())))
    else:
        with torch.no_grad():
            for i, (data, _) in enumerate(test_loader):
                data = data.to(device)
                pred = classifier(data, split_point=args.split_point)
                recon = decoder(pred)
                mse_sum += per_pixel_mse(recon, data) * data.size(0)
                snr_sum += per_sample_snr(data, recon) * data.size(0)
                n += data.size(0)
                if args.wer and args.dataset == 'cremad' and len(wer_items) < args.wer_subset:
                    stems = test_set.items
                    for j in range(min(data.size(0), args.wer_subset - len(wer_items))):
                        idx = i * 128 + j
                        if idx >= len(stems):
                            break
                        stem = os.path.basename(stems[idx]).replace('_mel.npy', '')
                        audio = recon_mel_to_audio(recon[j].squeeze().cpu().numpy())
                        hyp = whisper_model.transcribe(audio, language='en',
                                                       task='transcribe',
                                                       fp16=True)['text'].strip()
                        gt = SENTENCE_MAP[stem.split('_')[1]]
                        e = edist(norm(gt), norm(hyp))
                        wer_items.append((e, len(norm(gt).split())))
    mse = mse_sum / max(1, n)
    snr = snr_sum / max(1, n)
    print(f'[{args.dataset} {args.model} {args.mode} sp{args.split_point}] '
          f'test_mse={mse:.6f}  test_snr={snr:.2f}dB  n={n}')
    if wer_items:
        te = sum(e for e, _ in wer_items)
        tw = sum(w for _, w in wer_items)
        print(f'WER (GT->Recon, normalized, corpus, n={len(wer_items)}): '
              f'{100 * te / max(1, tw):.2f}%')


if __name__ == '__main__':
    main()
