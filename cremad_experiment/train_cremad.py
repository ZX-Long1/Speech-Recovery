import os, sys, json, argparse, time, random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from cremad_experiment.cremad_dataset import CremaDataset
from cremad_experiment.cremad_models import BasicBlock, ResNet18

def train(classifier, loader, optimizer, epoch, log_interval, device):
    classifier.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        if target.dtype != torch.long:
            target = target.long()
        optimizer.zero_grad()
        output = classifier(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(loader.dataset)}] Loss: {loss.item():.6f}')

def test(classifier, loader, device):
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            if target.dtype != torch.long:
                target = target.long()
            output = classifier(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(loader.dataset)
    acc = 100. * correct / len(loader.dataset)
    print(f'Test: avg loss={test_loss:.6f}, acc={correct}/{len(loader.dataset)} ({acc:.2f}%)')
    return acc / 100.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat', type=str, default='mel', choices=['mel', 'mfcc'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--cuda_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.cuda_id)
    print(f'Device: {device} | Feature: {args.feat}')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    transform = transforms.Compose([transforms.ToTensor()])
    base = '/media/sda1/zxlong/tmc_re/CREMA-D/processed'
    train_set = CremaDataset(f'{base}/train', args.feat, transform)
    test_set = CremaDataset(f'{base}/test', args.feat, transform)
    train_loader = torch.utils.data.DataLoader(train_set, args.batch_size, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=False)
    print(f'Train: {len(train_set)}, Test: {len(test_set)}')

    classifier = ResNet18(BasicBlock, nc=1, nz=6).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

    out_dir = f'/media/sda1/zxlong/tmc_re/CREMA-D/classifier/{args.feat}_train_record/'
    os.makedirs(out_dir, exist_ok=True)

    best_acc, best_epoch = 0, 0
    for epoch in range(1, args.epochs + 1):
        train(classifier, train_loader, optimizer, epoch, args.log_interval, device)
        acc = test(classifier, test_loader, device)

        with open(f'{out_dir}record_test.txt', 'a') as f:
            f.write(f'Epoch {epoch}: acc={acc:.6f}\n')

        if (epoch % 10) == 1:
            torch.save({'epoch': epoch, 'model': classifier.state_dict(), 'acc': acc}, f'{out_dir}classifier_{epoch}.pth')

        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            torch.save({'epoch': epoch, 'model': classifier.state_dict(), 'best_cl_acc': best_acc}, f'{out_dir}classifier.pth')

    print(f'Best: epoch {best_epoch}, acc {best_acc:.6f}')

if __name__ == '__main__':
    main()
