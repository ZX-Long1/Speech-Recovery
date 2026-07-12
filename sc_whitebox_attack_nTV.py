import os, sys, argparse, time, random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from speech_command_dataset1 import StudioSet
from speech_command_Model import BasicBlock, ResNet18, Conv_sp1, Conv_sp2, Conv_sp3, Conv_sp4
from utils import TV, speech_tf_loss
import matplotlib
matplotlib.use('Agg')

SC_BASE = '/media/sda1/zxlong/L_minghao/InverCRS/Speech_Command'
CLS_PATH = f'{SC_BASE}/Result/classifier/mel_spect_train_record/classifier.pth'
INV_CLS = {1: Conv_sp1, 2: Conv_sp2, 3: Conv_sp3, 4: Conv_sp4}
CACHE_PATH = '/tmp/sc_mel_dataset_cache.pt'

def load_dataset(transform):
    if os.path.exists(CACHE_PATH):
        print(f'Loading from cache: {CACHE_PATH}', flush=True)
        train_set, test_set = torch.load(CACHE_PATH)
        train_set.transform = transform
        test_set.transform = transform
        return train_set, test_set
    print('Loading raw dataset (no cache found)...', flush=True)
    train_set = StudioSet(root=f'{SC_BASE}/audio/train_set_mel_spect_numpy', transform=transform)
    test_set = StudioSet(root=f'{SC_BASE}/audio/test_set_mel_spect_numpy', transform=transform)
    return train_set, test_set

def train(classifier, inversion, loader, optimizer, epoch, sp, nTV, log_interval, device, regularizer='tvd', alpha=1.0):
    classifier.eval()
    inversion.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data, split_point=sp)
        reconstruction = inversion(prediction)
        recon_pred = classifier(reconstruction, split_point=sp)
        fm_loss = F.mse_loss(recon_pred, prediction)

        if regularizer == 'none':
            total_loss = fm_loss
            reg_loss = torch.tensor(0.0)
            temporal_loss = torch.tensor(0.0)
            spectral_loss = torch.tensor(0.0)
        elif regularizer == 'tvd':
            reg = TV(reconstruction)
            reg_loss = nTV * reg
            total_loss = fm_loss + reg_loss
            temporal_loss = reg
            spectral_loss = torch.tensor(0.0)
        else:  # speech_tf
            reg, t_loss, s_loss = speech_tf_loss(reconstruction, alpha)
            reg_loss = nTV * reg
            total_loss = fm_loss + reg_loss
            temporal_loss = t_loss
            spectral_loss = s_loss

        total_loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            n = batch_idx * len(data)
            total = len(loader.dataset)
            print(f'Epoch {epoch} [{n}/{total}] '
                  f'fm_loss={fm_loss.item():.6f} '
                  f'temporal={temporal_loss.item():.6f} '
                  f'spectral={spectral_loss.item():.6f} '
                  f'reg={reg_loss.item():.6f} '
                  f'total={total_loss.item():.6f}')

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
    mse /= len(loader.dataset) * 64 * 96
    return mse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split_point', type=int, default=1, choices=[1,2,3,4])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--nTV', type=float, default=0.0)
    parser.add_argument('--regularizer', type=str, default='tvd',
                        choices=['none', 'tvd', 'speech_tf'])
    parser.add_argument('--alpha', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.cuda_id)
    print(f'Device: {device}  sp={args.split_point}  reg={args.regularizer}  nTV={args.nTV}  alpha={args.alpha}', flush=True)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    transform = transforms.Compose([transforms.ToTensor()])
    train_set, test_set = load_dataset(transform)
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=False)
    print(f'train={len(train_set)}, test={len(test_set)}', flush=True)

    classifier = ResNet18(BasicBlock, nc=1, nz=35).to(device)
    ckpt = torch.load(CLS_PATH, map_location='cpu')
    classifier.load_state_dict(ckpt['model'])
    print(f'Loaded classifier (epoch={ckpt.get("epoch","?")}, acc={ckpt.get("best_cl_acc","?"):.6f})', flush=True)

    inversion = INV_CLS[args.split_point]().to(device)
    optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

    out_dir = f'{SC_BASE}/Result/whitebox_shallow_{args.regularizer}_nTV{args.nTV}/split_point{args.split_point}/'
    os.makedirs(f'{out_dir}train_record/', exist_ok=True)

    best_mse, best_epoch = float('inf'), 0
    for epoch in range(1, args.epochs + 1):
        train(classifier, inversion, train_loader, optimizer, epoch, args.split_point, args.nTV,
              args.log_interval, device, args.regularizer, args.alpha)
        mse = test(classifier, inversion, test_loader, args.split_point, device)
        with open(f'{out_dir}train_record/record_test.txt', 'a') as f:
            f.write(f'Epoch {epoch}: MSE={mse:.8f}\n')
        print(f'  Test MSE: {mse:.8f}', flush=True)

        if mse < best_mse:
            best_mse, best_epoch = mse, epoch
            torch.save({'best_epoch': epoch, 'model': inversion.state_dict(), 'best_acc': mse},
                       f'{out_dir}train_record/inversion.pth')

    print(f'Best: epoch {best_epoch}, MSE {best_mse:.8f}', flush=True)

if __name__ == '__main__':
    main()
