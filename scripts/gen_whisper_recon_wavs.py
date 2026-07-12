import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import whisper

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cremad_experiment.whisper_models import WhisperEncoder, Decoder_whitebox, Decoder_blackbox

BASE = '/media/sda1/zxlong/tmc_re/CREMA-D'
WAV_DIR = f'{BASE}/AudioWAV'
REF_DIR = f'{BASE}/eval_results/audio_samples/mel_blackbox'  # to get the 20 stems
OUT_BASE = f'{BASE}/eval_results/audio_samples'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
SPLIT_POINTS = [1, 2, 4, 8]
MODES = ['blackbox', 'whitebox']

def get_stems():
    import glob
    clean_files = sorted(glob.glob(f'{REF_DIR}/clean_sp1_*.wav'))
    stems = [os.path.basename(f).replace('clean_sp1_', '').replace('.wav', '') for f in clean_files]
    print(f'Found {len(stems)} stems', flush=True)
    return stems

def reconstruct_audio(decoder_output_np):
    power_mel = np.power(10.0, decoder_output_np)
    S = librosa.feature.inverse.mel_to_stft(power_mel, sr=16000, n_fft=400, power=2.0)
    audio = librosa.griffinlim(S, hop_length=160, n_iter=32)
    return audio

def main():
    stems = get_stems()
    print(f'Device: {DEVICE}', flush=True)

    print('Loading Whisper encoder...', flush=True)
    enc = WhisperEncoder(split_point=1, freeze=True).to(DEVICE)
    enc.eval()

    for mode in MODES:
        out_dir = f'{OUT_BASE}/whisper_{mode}'
        os.makedirs(out_dir, exist_ok=True)

        decoder_cls = Decoder_blackbox if mode == 'blackbox' else Decoder_whitebox

        # Save clean files once
        for stem in stems:
            clean_path = os.path.join(out_dir, f'clean_sp1_{stem}.wav')
            if not os.path.exists(clean_path):
                wav_path = os.path.join(WAV_DIR, f'{stem}.wav')
                audio_orig, _ = librosa.load(wav_path, sr=16000, mono=True)
                sf.write(clean_path, audio_orig, 16000)

        for sp in SPLIT_POINTS:
            enc.split_point = sp
            dec_path = f'{BASE}/whisper_inversion/sp{sp}_{mode}/decoder_best.pth'
            if not os.path.exists(dec_path):
                print(f'  [SKIP] sp={sp} {mode}: no checkpoint at {dec_path}', flush=True)
                continue

            print(f'  Loading decoder sp={sp} {mode}...', flush=True)
            decoder = decoder_cls().to(DEVICE)
            decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE))
            decoder.eval()

            for stem in stems:
                recon_path = os.path.join(out_dir, f'recon_sp{sp}_{stem}.wav')
                if os.path.exists(recon_path):
                    continue

                wav_path = os.path.join(WAV_DIR, f'{stem}.wav')
                audio, _ = librosa.load(wav_path, sr=16000, mono=True)

                mel = whisper.log_mel_spectrogram(audio, padding=0).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    feat = enc(mel)
                    recon = decoder(feat)

                t = min(recon.shape[-1], mel.shape[-1])
                recon_mel = recon[:, :, :t]
                recon_np = recon_mel.squeeze(0).cpu().numpy()
                recon_audio = reconstruct_audio(recon_np)
                sf.write(recon_path, recon_audio, 16000)

            print(f'    Done sp={sp} {mode}: {len(stems)} files', flush=True)

    print('All done.', flush=True)

if __name__ == '__main__':
    main()