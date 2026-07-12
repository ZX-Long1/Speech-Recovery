# baseline_igd_US_mfcc.py
from __future__ import print_function, division
import os
import argparse
import random
import csv
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
import librosa.display
from torchvision import transforms

# 引入项目数据集与模型（确保 Dataset2.StudioSet 与 Model2.Classifier 在 python path 中）
from Dataset2 import StudioSet
from Model2 import Classifier

DEFAULT_SR = 25250
DEFAULT_HOP_LENGTH = 512

# -----------------------
# 绘制 MFCC 图（原 vs IGD）
# -----------------------
def plot_and_save_mfcc(orig_tensor, rec_tensor, save_path, sr=DEFAULT_SR, hop_length=DEFAULT_HOP_LENGTH):
    orig = orig_tensor.cpu().numpy()
    rec = rec_tensor.cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    librosa.display.specshow(orig[0,0] if orig.ndim==4 else orig[0], sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel', cmap='magma')
    plt.title('Original MFCC (uint8 scaled)')
    plt.colorbar(format='%+2.0f')

    plt.subplot(1, 2, 2)
    librosa.display.specshow(rec[0,0] if rec.ndim==4 else rec[0], sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='mel', cmap='magma')
    plt.title('Recovered MFCC (IGD)')
    plt.colorbar(format='%+2.0f')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------
# IGD Baseline (same as Mel version)
# ---------------------------------------------------------
def IGD_recover_random(victim, x_sp_target, init_shape, device,
                       split_point=1,
                       lr=0.001,
                       iters=2000,
                       tv_weight=1e-4,
                       print_interval=1000,
                       init_from_real=False,
                       noise_scale=0.1):
    # init can be random or noisy constant
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
    # orig, rec: numpy arrays
    num = np.sum(orig ** 2)
    den = np.sum((orig - rec) ** 2) + eps
    return 10.0 * np.log10(num / den)

def compute_mse_from_tensor(orig, rec):
    return float(np.mean((orig - rec) ** 2))

# -----------------------
# 主函数
# -----------------------
def main():
    parser = argparse.ArgumentParser(description='IGD baseline multi-split CI-style (UrbanSound8K - MFCC)')
    parser.add_argument('--iters', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--tv', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--test_root', type=str,
                        default='../UrbanSound8K/UrbanSound8K/audio/test_set_mfccsnumpy')
    parser.add_argument('--classifier_path', type=str,
                        default='../UrbanSound8K/UrbanSound8K/classifier/train_record/classifier.pth')
    parser.add_argument('--target_dir', type=str, default='../baseline/baseline_US_mfcc/')
    parser.add_argument('--sr', type=int, default=DEFAULT_SR)
    parser.add_argument('--hop_length', type=int, default=DEFAULT_HOP_LENGTH)
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
    dataset = StudioSet(root=args.test_root, transform=transform)
    n_total = len(dataset)
    print("Test size:", n_total)
    indices = random.sample(range(n_total), min(args.num_samples, n_total))
    print("Selected idx =", indices)

    # 加载分类器（MFCC 版）
    classifier = Classifier(nc=1, ndf=64, nz=10).to(device)
    chk = torch.load(args.classifier_path, map_location=device)
    state = chk.get("model", chk)
    classifier.load_state_dict(state)
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
                # ensure shape: (1, C, H, W)
                if sample.dim() == 2:
                    sample = sample.unsqueeze(0)
                if sample.dim() == 3:
                    sample = sample.unsqueeze(0)

                mfcc_input = sample.to(device)
                with torch.no_grad():
                    target_activation = classifier(mfcc_input, split_point=split)

                rec_cpu = IGD_recover_random(
                    classifier, target_activation, mfcc_input.shape, device,
                    split_point=split, lr=args.lr, iters=args.iters,
                    tv_weight=args.tv, print_interval=max(1, args.iters // 10)
                )

                # -----------------------------
                # CI-style SNR & MSE 计算
                # -----------------------------
                orig_np = mfcc_input.cpu().detach().numpy()
                rec_np = rec_cpu.cpu().detach().numpy()

                snr = compute_snr_from_tensor(orig_np, rec_np)
                mse = compute_mse_from_tensor(orig_np, rec_np)
                snr_list.append(snr)
                mse_list.append(mse)

                print(f"idx {idx}, label={int(label)}, SNR={snr:.4f} dB, MSE={mse:.6e}")

                # 保存图
                fig_path = out_dir + f"fig/sample_{i}_idx{idx}_lbl{label}_snr{snr:.2f}.png"
                plot_and_save_mfcc(mfcc_input.cpu(), rec_cpu, fig_path, sr=args.sr, hop_length=args.hop_length)

                # 保存 numpy
                np.save(out_dir + f"res/orig_{idx}.npy", orig_np)
                np.save(out_dir + f"res/rec_{idx}.npy", rec_np)

                rows.append([idx, int(label), snr, mse])

            except Exception as e:
                print("Error on idx", idx, ":", e)
                traceback.print_exc()
                continue

        # 保存 CSV 和平均值
        if len(snr_list) == 0:
            avg_snr = float('nan')
            avg_mse = float('nan')
        else:
            avg_snr = float(np.mean(snr_list))
            avg_mse = float(np.mean(mse_list))
        rows.append(['average', '-', avg_snr, avg_mse])

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['sample_idx', 'label', 'snr_db', 'mse'])
            writer.writerows(rows)

        print(f"Split {split} done. Avg SNR={avg_snr:.4f} dB, Avg MSE={avg_mse:.6e}")
    print("All done.")

if __name__ == "__main__":
    main()
