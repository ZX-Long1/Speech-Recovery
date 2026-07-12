import os, sys, random, warnings
warnings.filterwarnings('ignore')
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from speech_command_Model import BasicBlock, ResNet18, Conv_sp1, Conv_sp2, Conv_sp3, Conv_sp4

SC_BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'
MEL_MIN_VAL = -80.00000381469727
MEL_MAX_VAL = 3.814697265625e-06
CLS_PATH = f'{SC_BASE}/Result/classifier/mel_spect_train_record/classifier.pth'
TEST_WAV = f'{SC_BASE}/audio/test_set'
TEST_MEL_NPY = f'{SC_BASE}/audio/test_set_mel_spect_numpy'
INV_CLS = {1: Conv_sp1, 2: Conv_sp2, 3: Conv_sp3, 4: Conv_sp4}
NUM_REC_SAVE = 100

def normalize(feat_db):
    norm = ((feat_db - MEL_MIN_VAL) / (MEL_MAX_VAL - MEL_MIN_VAL)) * 255
    return torch.from_numpy(np.uint8(norm)).unsqueeze(0).float() / 255.0

def denormalize(tensor):
    return tensor.squeeze().cpu().numpy() * (MEL_MAX_VAL - MEL_MIN_VAL) + MEL_MIN_VAL

def compute_snr(orig, recon):
    noise_pow = torch.mean((orig - recon) ** 2)
    signal_pow = torch.mean(orig ** 2)
    if noise_pow < 1e-12:
        return float('inf')
    return 10 * torch.log10(signal_pow / noise_pow).item()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--nTV', type=float, required=True)
    parser.add_argument('--regularizer', type=str, default='speech_tf', choices=['none', 'tvd', 'speech_tf'])
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}  reg={args.regularizer}  nTV={args.nTV}', flush=True)

    ntv_str = f'nTV{args.nTV}'
    if args.regularizer == 'none':
        OUTPUT_DIR = f'/media/sda1/zxlong/tmc_re/sc_eval_results_mel_{ntv_str}'
    elif args.regularizer == 'tvd':
        OUTPUT_DIR = f'/media/sda1/zxlong/tmc_re/sc_eval_results_mel_{ntv_str}'
    else:
        OUTPUT_DIR = f'/media/sda1/zxlong/tmc_re/sc_eval_results_mel_{args.regularizer}_{ntv_str}'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_files = sorted(os.listdir(TEST_WAV))
    rng = random.Random(42)
    rng.shuffle(wav_files)
    print(f'Test WAVs: {len(wav_files)}', flush=True)

    mel_cache = torch.load('/tmp/sc_mel_cache/test_mel_dict.pt', map_location='cpu')
    print(f'Loaded {len(mel_cache)} mel npy files from combined cache', flush=True)

    classifier = ResNet18(BasicBlock, 1, 35).to(device)
    ckpt = torch.load(CLS_PATH, map_location='cpu')
    classifier.load_state_dict(ckpt['model'])
    classifier.eval()

    all_rows = []
    mode = 'whitebox_shallow'
    split_points = [1, 2, 3, 4]

    for sp in split_points:
        print(f'\n{">"*20} Split Point {sp} {"<"*20}', flush=True)

        if args.regularizer == 'none':
            ckpt_dir = f'{SC_BASE}/Result/whitebox_shallow_{ntv_str}/split_point{sp}/train_record/'
        elif args.regularizer == 'tvd':
            ckpt_dir = f'{SC_BASE}/Result/whitebox_shallow_{ntv_str}/split_point{sp}/train_record/'
        else:
            ckpt_dir = f'{SC_BASE}/Result/whitebox_shallow_{args.regularizer}_{ntv_str}/split_point{sp}/train_record/'

        inversion = INV_CLS[sp]().to(device)
        ckpt_path = os.path.join(ckpt_dir, 'inversion.pth')
        if not os.path.exists(ckpt_path):
            print(f'  WARNING: checkpoint not found at {ckpt_path}', flush=True)
            continue
        ckpt = torch.load(ckpt_path, map_location='cpu')
        inversion.load_state_dict(ckpt['model'])
        inversion.eval()
        print(f'  Loaded inversion (epoch={ckpt.get("best_epoch","?")})', flush=True)

        os.makedirs(f'{OUTPUT_DIR}/reconstructed_mel/{mode}/sp{sp}', exist_ok=True)

        mse_list, snr_list = [], []
        ci_list, cr_list = [], []

        for idx, wav_name in enumerate(tqdm(wav_files, desc=f'sp{sp}')):
            label = int(wav_name.split('_')[0])
            feat_db = mel_cache[wav_name]
            inp = normalize(feat_db).unsqueeze(0).to(device)

            with torch.no_grad():
                logit_inp = classifier(inp, split_point=0)
                ft = classifier(inp, split_point=sp)
                rec = inversion(ft)
                logit_rec = classifier(rec, split_point=0)

            ci = int(logit_inp.argmax(dim=1).item() == label)
            cr = int(logit_rec.argmax(dim=1).item() == label)
            mse = F.mse_loss(rec, inp).item()
            snr = compute_snr(inp, rec)

            mse_list.append(mse)
            snr_list.append(snr)
            ci_list.append(ci)
            cr_list.append(cr)

            all_rows.append({
                'feature': 'mel', 'mode': mode, 'split_point': sp,
                'condition': 'clean',
                'sample': wav_name, 'label': label,
                'mse': mse, 'snr': snr,
                'correct_input': ci, 'correct_recon': cr,
            })

            if idx < NUM_REC_SAVE:
                rec_np = denormalize(rec)
                np.save(f'{OUTPUT_DIR}/reconstructed_mel/{mode}/sp{sp}/{wav_name.replace(".wav","")}.npy', rec_np)

        n = len(wav_files)
        acc_inp = np.mean(ci_list) * 100
        acc_rec = np.mean(cr_list) * 100
        print(f'  MSE={np.mean(mse_list):.6f}  SNR={np.mean(snr_list):.2f}  '
              f'ACC_inp={acc_inp:.1f}%  ACC_rec={acc_rec:.1f}%  drop={acc_inp-acc_rec:+.1f}%', flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(f'{OUTPUT_DIR}/eval_detail.csv', index=False)
    print(f'\nSaved {OUTPUT_DIR}/eval_detail.csv', flush=True)

    summary_rows = []
    for (feat, mode, sp, cond), grp in df.groupby(['feature', 'mode', 'split_point', 'condition']):
        summary_rows.append({
            'feature': feat, 'mode': mode, 'split_point': sp, 'condition': cond,
            'mse_mean': grp['mse'].mean(), 'mse_std': grp['mse'].std(),
            'snr_mean': grp['snr'].mean(), 'snr_std': grp['snr'].std(),
            'acc_input': grp['correct_input'].mean() * 100,
            'acc_recon': grp['correct_recon'].mean() * 100,
            'acc_drop': (grp['correct_input'].mean() - grp['correct_recon'].mean()) * 100,
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(f'{OUTPUT_DIR}/eval_summary.csv', index=False)
    print(f'Saved {OUTPUT_DIR}/eval_summary.csv', flush=True)
    print(df_summary.to_string(index=False))
    print('\nDone!', flush=True)

if __name__ == '__main__':
    main()
