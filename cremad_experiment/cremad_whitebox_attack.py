import os, sys, json, argparse, time, random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cremad_experiment.cremad_dataset import CremaDataset
from cremad_experiment.cremad_models import BasicBlock, ResNet18
from cremad_experiment.cremad_models import Conv_sp1, Conv_sp2, Conv_sp3, Conv_sp4
from cremad_experiment.cremad_models import Conv_sp1_mfcc, Conv_sp2_mfcc, Conv_sp3_mfcc, Conv_sp4_mfcc
from cremad_experiment.cremad_models import Conv_sp1_shallow, Conv_sp2_shallow, Conv_sp3_shallow, Conv_sp4_shallow

def TV(x):
    batch_size = x.size(0)
    h_x, w_x = x.size(2), x.size(3)
    count_h = x[:, :, 1:, :].numel()
    count_w = x[:, :, :, 1:].numel()
    h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x-1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size

INV_CLS = {1: Conv_sp1, 2: Conv_sp2, 3: Conv_sp3, 4: Conv_sp4}
INV_CLS_MFCC = {1: Conv_sp1_mfcc, 2: Conv_sp2_mfcc, 3: Conv_sp3_mfcc, 4: Conv_sp4_mfcc}
INV_CLS_SHALLOW = {1: Conv_sp1_shallow, 2: Conv_sp2_shallow, 3: Conv_sp3_shallow, 4: Conv_sp4_shallow}

def train(classifier, inversion, loader, optimizer, epoch, sp, nTV, log_interval, device):
    classifier.eval()
    inversion.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data, split_point=sp)
        reconstruction = inversion(prediction)
        recon_pred = classifier(reconstruction, split_point=sp)
        loss = F.mse_loss(recon_pred, prediction) + nTV * TV(reconstruction)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch {epoch} [{batch_idx*len(data)}/{len(loader.dataset)}] Loss: {loss.item():.8f}')

def test(classifier, inversion, loader, sp, device):
    classifier.eval()
    inversion.eval()
    mse = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, split_point=sp)
            reconstruction = inversion(prediction)
            mse += F.mse_loss(reconstruction, data, reduction='sum').item()
    mse /= len(loader.dataset) * 64 * 640  # normalized by pixels
    return mse

def record(classifier, inversion, loader, epoch, sp, nTV, out_dir, device):
    classifier.eval()
    inversion.eval()
    os.makedirs(f'{out_dir}res/epoch{epoch}/', exist_ok=True)
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, split_point=sp)
            reconstruction = inversion(prediction)
            num = min(8, data.size(0))
            for i in range(num):
                truth, inv = data[i].cpu(), reconstruction[i].cpu()
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                for ax, img, title in zip(axes, [truth, inv], ['Original', 'Reconstruction']):
                    arr = img.squeeze(0).numpy()
                    arr = arr * 255  # roughly undo norm for vis
                    librosa.display.specshow(arr, sr=16000, hop_length=80,
                                             x_axis='time', y_axis='mel', ax=ax, cmap='magma')
                    ax.set_title(title)
                plt.tight_layout()
                plt.savefig(f'{out_dir}res/epoch{epoch}/sample{i}.png', dpi=150, bbox_inches='tight')
                plt.close()
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat', type=str, default='mel', choices=['mel', 'mfcc'])
    parser.add_argument('--split_point', type=int, default=1, choices=[1,2,3,4])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--nTV', type=float, default=0.5)
    parser.add_argument('--aux_set', action='store_true', default=True)
    parser.add_argument('--shallow', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.cuda_id)
    print(f'Device: {device}  feat={args.feat}  sp={args.split_point}')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    transform = transforms.Compose([transforms.ToTensor()])
    base = '/media/sda1/zxlong/tmc_re/CREMA-D/processed'
    train_set = CremaDataset(f'{base}/aux', args.feat, transform)
    test_set = CremaDataset(f'{base}/test', args.feat, transform)
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=False)
    print(f'aux={len(train_set)}, test={len(test_set)}')

    classifier = ResNet18(BasicBlock, nc=1, nz=6).to(device)
    cl_path = f'/media/sda1/zxlong/tmc_re/CREMA-D/classifier/{args.feat}_train_record/classifier.pth'
    ckpt = torch.load(cl_path, map_location='cpu')
    classifier.load_state_dict(ckpt['model'])
    print(f'Loaded classifier (epoch={ckpt.get("epoch","?")}, acc={ckpt.get("best_cl_acc","?"):.6f})')

    if args.shallow:
        inv_map = INV_CLS_SHALLOW
    elif args.feat == 'mfcc':
        inv_map = INV_CLS_MFCC
    else:
        inv_map = INV_CLS
    inversion = inv_map[args.split_point]().to(device)
    optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

    wb_dir = 'whitebox_shallow' if args.shallow else 'whitebox'
    out_dir = f'/media/sda1/zxlong/tmc_re/CREMA-D/inversion/{args.feat}/{wb_dir}/split_point{args.split_point}/'
    os.makedirs(f'{out_dir}res/', exist_ok=True)
    os.makedirs(f'{out_dir}train_record/', exist_ok=True)

    best_mse, best_epoch = float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        train(classifier, inversion, train_loader, optimizer, epoch, args.split_point, args.nTV, args.log_interval, device)
        mse = test(classifier, inversion, test_loader, args.split_point, device)
        with open(f'{out_dir}train_record/record_test.txt', 'a') as f:
            f.write(f'Epoch {epoch}: MSE={mse:.8f}\n')
        print(f'  Test MSE: {mse:.8f}')

        if epoch % 10 == 0 or epoch == 1:
            record(classifier, inversion, test_loader, epoch, args.split_point, args.nTV, out_dir, device)

        if mse < best_mse:
            best_mse, best_epoch = mse, epoch
            torch.save({'best_epoch': epoch, 'model': inversion.state_dict(), 'best_acc': mse},
                       f'{out_dir}train_record/inversion.pth')

    print(f'Best: epoch {best_epoch}, MSE {best_mse:.8f}')

if __name__ == '__main__':
    main()
