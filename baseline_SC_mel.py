# baseline_igd_speechcommand_mel.py
from __future__ import print_function, division
import os
import argparse
import random
import csv
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import librosa.display
from torchvision import transforms

# SpeechCommand dataset & model (ensure these modules exist in your project)
from speech_command_dataset1 import StudioSet as StudioSet1
from speech_command_Model import BasicBlock, ResNet18, Conv_sp1, Conv_sp2, Conv_sp3, Conv_sp4

DEFAULT_SR = 49050
DEFAULT_HOP_LENGTH = 512

# -----------------------
# 绘制 Mel 图（原 vs IGD）
# -----------------------
def plot_and_save_mel(orig_tensor, rec_tensor, save_path, sr=DEFAULT_SR, hop_length=DEFAULT_HOP_LENGTH):
    orig = orig_tensor.cpu().numpy()
    rec = rec_tensor.cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(orig[0,0] if orig.ndim==4 else orig[0], sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel', cmap='magma')
    plt.title('Original Mel (SpeechCommand)')
    plt.colorbar(format='%+2.0f')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(rec[0,0] if rec.ndim==4 else rec[0], sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel', cmap='magma')
    plt.title('Recovered Mel (IGD)')
    plt.colorbar(format='%+2.0f')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# IGD Baseline
# ---------------------------------------------------------
def IGD_recover_random(victim, x_sp_target, init_shape, device,
                       split_point=1,
                       lr=0.001,
                       iters=2000,
                       tv_weight=1e-4,
                       print_interval=1000,
                       init_from_real=False,
                       noise_scale=0.1):
    if init_from_real:
        x = torch.ones(init_shape, device=device) * 0.5
        x = x + torch.randn_like(x) * noise_scale
    else:
        x = torch.rand(init_shape, device=device)

    x = x.clamp(0.0, 1.0)
    x.requires_grad_(True)

    optimizer = optim.Adam([x], lr=lr)
    mse_loss = nn.MSELoss()

    def tv_loss(t):
        dh = torch.mean(torch.abs(t[:, :, 1:, :] - t[:, :, :-1, :]))
        dw = torch.mean(torch.abs(t[:, :, :, 1:] - t[:, :, :, :-1]))
        return dh + dw

    for it in range(1, iters + 1):
        optimizer.zero_grad()
        x_sp = victim(x, split_point=split_point)
        loss = mse_loss(x_sp, x_sp_target)
        if tv_weight > 0:
            loss = loss + tv_weight * tv_loss(x)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0.0, 1.0)
        if it % print_interval == 0 or it == 1 or it == iters:
            print(f"[IGD Baseline] iter {it}/{iters} loss={loss.item():.8f}")

    return x.detach().cpu()

# -----------------------
# SNR & MSE（CI 原始方式）
# -----------------------
def compute_snr_from_tensor(orig, rec, eps=1e-30):
    num = np.sum(orig ** 2)
    den = np.sum((orig - rec) ** 2) + eps
    return 10.0 * np.log10(num / den)

def compute_mse_from_tensor(orig, rec):
    return float(np.mean((orig - rec) ** 2))

# -----------------------
# 主函数
# -----------------------
def main():
    parser = argparse.ArgumentParser(description='IGD baseline multi-split CI-style (SpeechCommand - Mel)')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--tv', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--cuda_id', type=int, default=1)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--test_root', type=str,
                        default='../Speech_Command/audio/test_set_mel_spect_numpy')
    parser.add_argument('--classifier_path', type=str,
                        default='../Speech_Command/Result/classifier/mel_spect_train_record/classifier.pth')
    parser.add_argument('--target_dir', type=str, default='../baseline/baseline_SC_mel/')
    parser.add_argument('--sr', type=int, default=DEFAULT_SR)
    parser.add_argument('--hop_length', type=int, default=DEFAULT_HOP_LENGTH)
    parser.add_argument('--nz', type=int, default=35)
    args = parser.parse_args()

    print("================================")
    print(args)
    print("================================")

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda_id}" if use_cuda else 'cpu')
    if use_cuda:
        torch.cuda.set_device(args.cuda_id)
    print("Device =", device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.path.exists(args.test_root):
        raise FileNotFoundError(args.test_root)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = StudioSet1(root=args.test_root, transform=transform)
    n_total = len(dataset)
    print("Test size:", n_total)
    indices = random.sample(range(n_total), min(args.num_samples, n_total))
    print("Selected idx =", indices)

    # 加载 ResNet18 分类器（SpeechCommand - mel）
    classifier = ResNet18(BasicBlock=BasicBlock, nc=1, nz=args.nz).to(device)
    chk = torch.load(args.classifier_path, map_location=device)
    state = chk.get("model", chk)
    try:
        classifier.load_state_dict(state)
    except Exception:
        # maybe checkpoint is direct state_dict
        classifier.load_state_dict(chk)
    classifier.eval()
    print("Loaded classifier.")

    split_points = [1, 2, 3, 4]

    for split in split_points:
        print(f"\n================ Split Point {split} ================")
        out_dir = args.target_dir.rstrip('/') + f'/split{split}/'
        os.makedirs(out_dir + 'fig/', exist_ok=True)
        os.makedirs(out_dir + 'res/', exist_ok=True)
        csv_path = out_dir + 'igd_baseline_results.csv'

        rows = []
        snr_list = []
        mse_list = []

        for i, idx in enumerate(indices):
            print("\n-----------------------------")
            print(f"Processing [{i+1}/{len(indices)}], dataset idx = {idx}")
            try:
                sample, label = dataset[idx]
                if not torch.is_tensor(sample):
                    sample = transform(sample)
                if sample.dim() == 2:
                    sample = sample.unsqueeze(0)
                if sample.dim() == 3:
                    sample = sample.unsqueeze(0)

                mel_input = sample.to(device)
                with torch.no_grad():
                    target_activation = classifier(mel_input, split_point=split)

                rec_cpu = IGD_recover_random(
                    classifier, target_activation, mel_input.shape, device,
                    split_point=split, lr=args.lr, iters=args.iters,
                    tv_weight=args.tv, print_interval=max(1, args.iters // 10)
                )

                # CI-style SNR & MSE 计算
                orig_np = mel_input.cpu().detach().numpy()
                rec_np = rec_cpu.cpu().detach().numpy()

                snr = compute_snr_from_tensor(orig_np, rec_np)
                mse = compute_mse_from_tensor(orig_np, rec_np)
                snr_list.append(snr)
                mse_list.append(mse)

                print(f"idx {idx}, label={int(label)}, SNR={snr:.4f} dB, MSE={mse:.6e}")

                # 保存图
                fig_path = out_dir + f"fig/sample_{i}_idx{idx}_lbl{label}_snr{snr:.2f}.png"
                plot_and_save_mel(mel_input.cpu(), rec_cpu, fig_path, sr=args.sr, hop_length=args.hop_length)

                # 保存 numpy
                np.save(out_dir + f"res/orig_{idx}.npy", orig_np)
                np.save(out_dir + f"res/rec_{idx}.npy", rec_np)

                rows.append([idx, int(label), snr, mse])

            except Exception as e:
                print("Error on idx", idx, ":", e)
                traceback.print_exc()
                continue

        # 保存 CSV 和平均值
        avg_snr = float(np.mean(snr_list)) if len(snr_list) > 0 else float('nan')
        avg_mse = float(np.mean(mse_list)) if len(mse_list) > 0 else float('nan')
        rows.append(['average', '-', avg_snr, avg_mse])

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_idx', 'label', 'snr_db', 'mse'])
            writer.writerows(rows)

        print(f"Split {split} done. Avg SNR={avg_snr:.4f} dB, Avg MSE={avg_mse:.6e}")
    print("All done.")

if __name__ == "__main__":
    main()
