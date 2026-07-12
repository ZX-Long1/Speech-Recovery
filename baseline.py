import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import random
import csv
import torch
from torch.utils.data import DataLoader


# ===========================================================
# IGD: Independent Gradient Descent Recovery
# ===========================================================
def IGD_recover(victim, x_sp_target, init_shape, split_point=1,
                lr=0.01, iters=3000):

    x = torch.randn(init_shape, device=x_sp_target.device, requires_grad=True)
    optimizer = optim.Adam([x], lr=lr)
    mse = nn.MSELoss()

    for i in range(iters):
        optimizer.zero_grad()

        x_sp = victim(x, split_point=split_point)

        loss = mse(x_sp, x_sp_target)

        loss.backward()
        optimizer.step()

        x.data = torch.clamp(x.data, -3, 3)

    return x.detach()


# ===========================================================
# Metrics
# ===========================================================
def compute_snr(x, x_rec):
    x = x.flatten()
    x_rec = x_rec.flatten()
    num = np.sum(x ** 2)
    den = np.sum((x - x_rec) ** 2) + 1e-8
    return 10 * np.log10(num / den)


def compute_sme(x, x_rec):
    return np.sum(np.abs(x - x_rec)) / (np.sum(np.abs(x)) + 1e-8)


# ===========================================================
# Plotting
# ===========================================================
def plot_original_and_recovered(x_orig, x_rec, save_path):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    librosa.display.specshow(x_orig.squeeze(), x_axis='time', y_axis='mel')
    plt.title("Original Mel")

    plt.subplot(1, 2, 2)
    librosa.display.specshow(x_rec.squeeze(), x_axis='time', y_axis='mel')
    plt.title("Reconstructed Mel (IGD)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ===========================================================
#  IGD Baseline Experiment
# ===========================================================

def run_baseline(split_point=1, model_path="./classifier_sp1.pth"):

    save_dir = f"./baseline/fig/sp{split_point}"
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------
    # Load dataset
    # ------------------------------
    dataset = StudioDataset(mode="test")
    print(f"Dataset loaded: {len(dataset)} test samples.")

    # ------------------------------
    # Load victim classifier
    # ------------------------------
    victim = Classifier(ndf=64, nc=1, dropout=0.3)
    victim.load_state_dict(torch.load(model_path))
    victim.cuda()
    victim.eval()
    print(f"Loaded victim model from: {model_path}")

    # ------------------------------
    # Randomly sample 20 examples
    # ------------------------------
    idx_list = random.sample(range(len(dataset)), 20)

    results = []
    csv_path = f"./baseline/igd_results_sp{split_point}.csv"

    for i, idx in enumerate(idx_list):
        mel, label = dataset[idx]
        mel = mel.unsqueeze(0).cuda()  # shape (1,1,64,200)

        # ------------------------------
        # Extract victim intermediate layer
        # ------------------------------
        with torch.no_grad():
            x_sp_target = victim(mel, split_point)

        # ------------------------------
        # IGD reconstruction
        # ------------------------------
        x_rec = IGD_recover(
            victim,
            x_sp_target,
            init_shape=mel.shape,
            split_point=split_point,
            lr=0.01,
            iters=3000
        ).cpu()

        # ------------------------------
        # Plot comparison
        ------------------------------
        save_path = os.path.join(save_dir, f"sample_{i}.png")
        plot_original_and_recovered(mel.cpu().numpy(), x_rec.numpy(), save_path)
        print(f"Saved figure: {save_path}")

        # ------------------------------
        # Metrics
        ------------------------------
        snr = compute_snr(mel.cpu().numpy(), x_rec.numpy())
        sme = compute_sme(mel.cpu().numpy(), x_rec.numpy())

        results.append([idx, label, snr, sme])

    # ------------------------------
    # Save CSV results
    # ------------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SampleID", "Class", "SNR", "SME"])
        writer.writerows(results)

    print(f"Baseline results saved → {csv_path}")


# ===========================================================
#  Entry
# ===========================================================
if __name__ == "__main__":
    run_baseline(split_point=1, model_path="./classifier_sp1.pth")
