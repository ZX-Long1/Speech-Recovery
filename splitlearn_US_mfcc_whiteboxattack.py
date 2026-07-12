from __future__ import print_function
import argparse                      #用于解析命令行参数
import torch                         #用于构建和训练神经网络
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F      #用于图像预处理
from Dataset2 import StudioSet       #数据集
from splitlearn_US_Model import Classifier, Conv_sp4, Conv_sp3, Conv_sp2, Conv_sp1  # 分类器，反转模型
import numpy as np
import random
import os, shutil                    #用于文件和目录操作
from utils import TV
import time
import librosa  #用于音频处理和特征提取
import librosa.display  # 用于音频特征的可视化
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=32, metavar='')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='')
parser.add_argument('--epochs', type=int, default=100, metavar='')
parser.add_argument('--lr', type=float, default=0.0002, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=20, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nz', type=int, default=10)
parser.add_argument('--num_workers', type=int, default=50, metavar='')
parser.add_argument('--select_layer', type=int, default=0, metavar='')
parser.add_argument('--frac', type=int, default=1, metavar='')
parser.add_argument('--split_point', type=int, default=1, metavar='')
parser.add_argument('--cuda_id', type=int, default=5, metavar='')
parser.add_argument('--nTV', type=float, default=1, metavar='')
parser.add_argument('--model_epoch', type=int, default=0, metavar='')


def train(classifier, inversion, device, data_loader, optimizer, epoch, split_point, nTV):
    classifier.eval()  #评估模式
    inversion.train()  #训练模式

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()  #在每次迭代开始前，清零梯度。PyTorch会累积梯度，因此需要初始化
        with torch.no_grad():
            prediction = classifier(data, split_point=split_point)

        #得到原始数据在分割点的特征x_sp，尝试恢复原始数据x0',然后再计算x0'在相同分割点的特征x'_sp
        reconstruction = inversion(prediction)
        reconstruction_prediction = classifier(reconstruction, split_point=split_point)

        loss_mse = F.mse_loss(reconstruction_prediction, prediction)
        loss = loss_mse + nTV * TV(reconstruction)
        loss.backward()
        optimizer.step()

        # 日志记录如果当前批次索引是log_interval的倍数，则打印当前的训练轮次、已处理的样本数、总样本数和当前损失。
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.8f}'.format(epoch, batch_idx * len(data),
                                                                 len(data_loader.dataset), loss.item()))

def test(classifier, inversion, device, data_loader, split_point, target_dir):
    classifier.eval()
    inversion.eval()
    mse_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, split_point=split_point)
            reconstruction = inversion(prediction)
            # print('data')
            # print(data)
            # print('reconstruction')
            # print(reconstruction)

            #计算平均均方误差MSE
            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()   #比较恢复的数据和原始数据的差别

    #将损失平均到每个像素，得到能表示整体重建质量的单个数值
    mse_loss /= (len(data_loader.dataset) * 64 * 200 * 1)

    with open(target_dir+'train_record/record_test.txt', 'a') as file:
        file.write('\nTest inversion: Average MSE loss: {:.8f},\n'.format(mse_loss))
    print('\nTest inversion: Average MSE loss: {:.8f},\n'.format(mse_loss))
    return mse_loss

def record(classifier, inversion, device, data_loader, epoch, num, target_dir, loss, split_point):
    #num:指定每个批次要处理的图像数量  loss:当前训练周期的损失值
    classifier.eval()
    inversion.eval()

    #不同训练轮次的图像分开存储
    os.makedirs(target_dir + 'res/epoch{}/'.format(epoch), exist_ok=True)

    with (torch.no_grad()):
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, split_point=split_point)
            reconstruction = inversion(prediction)

            #使用每个批次的前num个数据
            truth = data[0:num]
            inverse = reconstruction[0:num]

            max_val = 302.4324184398555
            min_val = -906.176513671875

            for i in range(num): #num=24
                # 原始图像
                # 将张量从 CUDA 移动到 CPU
                mfccs1 = truth[i].cpu()
                # 去掉第一个维度
                mfccs1 = mfccs1.squeeze(0)
                # print(mfccs1.shape)

                # 转换为 NumPy 数组
                mfccs1_numpy = mfccs1.numpy()
                # print(mfccs_numpy1.shape)

                #恢复原始音频参数
                mfccs1_numpy=mfccs1_numpy*((max_val-min_val))+min_val

                # 可视化MFCC特征
                sr = 25250
                plt.figure(figsize=(8, 4))
                librosa.display.specshow(mfccs1_numpy, sr=sr, x_axis='time', y_axis='mel', hop_length=512, cmap='magma')
                plt.colorbar(format='%+2.0f dB')
                plt.title('MFCC')
                plt.xlabel('Time (s)')
                plt.ylabel('MFCC Frequency')
                plt.tight_layout()
                # plt.show()
                plt.savefig(target_dir + 'res/epoch{}/origin_{}_{}_{:.8f}.png'.format(epoch, epoch, i, loss), dpi=300,bbox_inches='tight')  # 保存为高质量图片

                # 反转生成图像
                mfccs2 = inverse[i].cpu()
                mfccs2 = mfccs2.squeeze(0)
                mfccs2_numpy = mfccs2.numpy()
                mfccs2_numpy = mfccs2_numpy * ((max_val - min_val)) + min_val

                sr = 25250
                plt.figure(figsize=(8, 4))
                librosa.display.specshow(mfccs2_numpy, sr=sr, x_axis='time', y_axis='mel', hop_length=512, cmap='magma')
                plt.colorbar(format='%+2.0f dB')
                plt.title('MFCC')
                plt.xlabel('Time (s)')
                plt.ylabel('MFCC Frequency')
                plt.tight_layout()
                plt.savefig(target_dir + 'res/epoch{}/inverse_{}_{}_{:.8f}.png'.format(epoch,epoch, i, loss), dpi=300,bbox_inches='tight')  # 保存为高质量图片
                plt.close()  # 关闭图形对象以释放内存

            #确保只处理数据第一个批次然后退出循环
            break


def main():
    args = parser.parse_args()
    print("================================")
    print(args)
    print("================================")

    # target_dir = '../Speech_Command/Result/mel_spect_inversion/split_point' + str(args.split_point)+ '/'
    target_dir_root = '../SplitLearn/UrbanSound8K/mfcc/whitebox_inversion/split_point'

    #设置是否使用cuda
    cuda = "cuda:"+str(args.cuda_id)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(cuda if use_cuda else "cpu")
    torch.cuda.set_device(args.cuda_id)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if use_cuda else {}

    # 打印设备信息
    print(f"Selected device: {device}")
    print("Default device ID:", torch.cuda.current_device())  # 应与 args.cuda_id 一致
    print("Default device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    #设置随机数种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = StudioSet(root="../UrbanSound8K/UrbanSound8K/audio/train_set_mfccsnumpy", transform=transform)
    test_set = StudioSet(root="../UrbanSound8K/UrbanSound8K/audio/test_set_mfccsnumpy", transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    for i in [1,4]:
        args.split_point = i
        # target_dir = target_dir_root + str(args.split_point) + '/'
        for j in [0,5,10,15,20,25,50,100]:
            # target_dir=target_dir + 'inversion_'+str(j)+'/'
            target_dir = target_dir_root + str(args.split_point) + '/' + 'inversion_' + str(j) + '/'
            os.makedirs(target_dir, exist_ok=True)
            os.makedirs(target_dir+'res/', exist_ok=True)
            os.makedirs(target_dir+'train_record/', exist_ok=True)

            # 从指定的路径加载一个训练好的模型及其训练信息
            classifier = Classifier(nc=args.nc, ndf=args.ndf, nz=args.nz).to(device)
            path = '../SplitLearn/UrbanSound8K/mfcc/classifier/classifier_' + str(j) + '.pth'
            str1 = 'epoch'
            str2 = 'acc'
            print(path)
            try:
                checkpoint = torch.load(path, map_location={'cuda:'+str(args.cuda_id): cuda})
                classifier.load_state_dict(checkpoint['model'])
                best_epoch = checkpoint[str1]
                best_acc = checkpoint[str2]
                print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.8f})".format(path, best_epoch, best_acc))
            except:
                print("=> load classifier checkpoint '{}' failed".format(path))
                return

            if args.split_point == 1:
                inversion = Conv_sp1().to(device)
            elif args.split_point == 2:
                inversion = Conv_sp2().to(device)
            elif args.split_point == 3:
                inversion = Conv_sp3().to(device)
            elif args.split_point == 4:
                inversion = Conv_sp4().to(device)

            optimizer = optim.Adam(inversion.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)   #原始学习率

            best_acc = 1
            best_epoch = 0
            start_time = time.time()

            # Train classifier
            for epoch in range(1, args.epochs + 1):
                train(classifier, inversion, device, train_loader, optimizer, epoch, args.split_point, args.nTV)
                cl_acc = test(classifier, inversion, device, test_loader, args.split_point, target_dir)

                with open(target_dir+'train_record/record_time.txt', 'a') as file:
                    file.write('\nEpoch:{} -- Time Consume: {:.6f},\n'.format(epoch, time.time()-start_time))

                if cl_acc < best_acc:
                    print('updata')
                    best_acc = cl_acc
                    best_epoch = epoch
                    state = {
                        'best_epoch': best_epoch,
                        'model': inversion.state_dict(),
                        'best_acc': best_acc,
                    }
                    record(classifier, inversion, device, test_loader, epoch,32, target_dir, cl_acc, args.split_point)
                    torch.save(state, target_dir+'train_record/inversion.pth')

            print("Best Inversion: epoch {}, acc {:.8f}".format(best_epoch, best_acc))
            with open(target_dir+'train_record/best_inversion.txt', 'w') as f:
                f.write("Best Inversion: epoch {}, acc {:.8f}\n".format(best_epoch, best_acc))


if __name__ == '__main__':
    main()