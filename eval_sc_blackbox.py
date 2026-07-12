import os, sys, argparse, time, torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
sys.path.insert(0, os.path.dirname(__file__))
from speech_command_dataset1 import StudioSet
from speech_command_Model import ResNet18, BasicBlock, Conv_sp1, Conv_sp2, Conv_sp3, Conv_sp4

SC_BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'
CLS_PATH = f'{SC_BASE}/Result/classifier/mel_spect_train_record/classifier.pth'
INV_CLS = {1: Conv_sp1, 2: Conv_sp2, 3: Conv_sp3, 4: Conv_sp4}
BATCH_SIZE = 256

def get_loader():
    transform = transforms.Compose([transforms.ToTensor()])
    return torch.utils.data.DataLoader(
        StudioSet(root=f'{SC_BASE}/audio/test_set_mel_spect_numpy', transform=transform),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

def get_classifier(device):
    model = ResNet18(BasicBlock, nc=1, nz=35).to(device)
    ck = torch.load(CLS_PATH, map_location='cpu')
    state = {k.replace('module.', ''): v for k, v in ck['model'].items()}
    model.load_state_dict(state)
    model.eval()
    return model

def evaluate(loader, classifier, inversion, sp, device):
    classifier.eval()
    inversion.eval()
    mse_sum = snr_sum = 0.0
    correct_orig = correct_recon = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            feat = classifier(data, split_point=sp)
            recon = inversion(feat)
            pred_orig = classifier(data, split_point=5)
            pred_recon = classifier(recon, split_point=5)

            b = data.size(0)
            mse = F.mse_loss(recon, data, reduction='none').view(b, -1).mean(dim=1)
            data_flat = data.view(b, -1)
            noise_flat = (data - recon).view(b, -1)
            var_signal = data_flat.var(dim=1) + 1e-12
            var_noise = noise_flat.var(dim=1) + 1e-12
            snr = 10 * torch.log10(var_signal / var_noise)

            mse_sum += mse.sum().item()
            snr_sum += snr.sum().item()
            correct_orig += (pred_orig.argmax(dim=1) == target).sum().item()
            correct_recon += (pred_recon.argmax(dim=1) == target).sum().item()
            total += b

    mse = mse_sum / total
    snr = snr_sum / total
    acc_orig = correct_orig / total
    acc_recon = correct_recon / total
    acc_drop = acc_orig - acc_recon
    return mse, snr, acc_orig, acc_recon, acc_drop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.cuda_id)
    print(f'Device: {device}', flush=True)

    loader = get_loader()
    print(f'Test set: {len(loader.dataset)} samples', flush=True)

    classifier = get_classifier(device)

    experiments = []
    for ns in [10, 20, 30, 40]:
        for sp in [1, 3]:
            pth = f'{SC_BASE}/Result/blackbox_mel_spect_inversion_ns{ns}/split_point{sp}/train_record/inversion.pth'
            out = f'{SC_BASE}/Result/blackbox_mel_spect_inversion_ns{ns}/split_point{sp}/train_record/eval_metrics.txt'
            experiments.append(('ablation', ns, sp, pth, out))
    for sp in [1, 3]:
        pth = f'{SC_BASE}/Result/blackbox_shallow_nTV0.5/split_point{sp}/train_record/inversion.pth'
        out = f'{SC_BASE}/Result/blackbox_shallow_nTV0.5/split_point{sp}/train_record/eval_metrics.txt'
        experiments.append(('baseline', 50, sp, pth, out))

    results = []
    for kind, ns, sp, pth_path, out_path in experiments:
        if not os.path.exists(pth_path):
            print(f'[SKIP] {kind} ns={ns} sp={sp} — checkpoint not found', flush=True)
            continue
        out_dir = os.path.dirname(out_path)
        if os.path.exists(out_path) and not args.force:
            with open(out_path) as f:
                line = f.read().strip()
            print(f'[CACHED] {kind} ns={ns} sp={sp}: {line}', flush=True)
            parts = [x.split('=')[1] for x in line.split(', ')]
            results.append((kind, ns, sp, float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
            continue

        print(f'[EVAL] {kind} ns={ns} sp={sp}', flush=True)
        inversion = INV_CLS[sp]().to(device)
        ck = torch.load(pth_path, map_location='cpu')
        inversion.load_state_dict(ck['model'])
        inversion.eval()
        t0 = time.time()
        mse, snr, acc_orig, acc_recon, acc_drop = evaluate(loader, classifier, inversion, sp, device)
        elapsed = time.time() - t0
        line = f'mse={mse:.8f}, snr={snr:.4f}, acc_orig={acc_orig:.6f}, acc_recon={acc_recon:.6f}, acc_drop={acc_drop:.6f}'
        os.makedirs(out_dir, exist_ok=True)
        with open(out_path, 'w') as f:
            f.write(line + '\n')
        print(f'  {line} ({elapsed:.1f}s)', flush=True)
        results.append((kind, ns, sp, mse, snr, acc_orig, acc_recon, acc_drop))

    print()
    header = f'{"kind":<12} {"ns":<4} {"sp":<4} {"MSE":<12} {"SNR(dB)":<12} {"Orig Acc":<12} {"Recon Acc":<12} {"Acc Drop":<12}'
    print(header)
    print('-' * len(header))
    for kind, ns, sp, mse, snr, acc_orig, acc_recon, acc_drop in results:
        label = f'{kind}_ns{ns}'
        print(f'{label:<12} {ns:<4} {sp:<4} {mse:<12.8f} {snr:<12.4f} {acc_orig:<12.6f} {acc_recon:<12.6f} {acc_drop:<12.6f}')

if __name__ == '__main__':
    main()