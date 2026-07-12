from __future__ import print_function
import argparse                      #用于解析命令行参数
import torch                         #用于构建和训练神经网络
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F      #用于图像预处理
# from Dataset1 import StudioSet as StudioSet1    #数据集
from speech_command_dataset1 import StudioSet as StudioSet1 #数据集
from speech_command_Model import BasicBlock,ResNet18,Conv_sp4, Conv_sp3, Conv_sp2, Conv_sp1  # 分类器，反转模型
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 以test_images作为训练集
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=50, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nz', type=int, default=35)
parser.add_argument('--num_workers', type=int, default=50, metavar='')
parser.add_argument('--split_point', type=int, default=1, metavar='')
parser.add_argument('--cuda_id', type=int, default=1,metavar='')
parser.add_argument('--noise', type=int, default=2, metavar='')


def classifier_acc(data_loader,args,device,cuda,target_dir):
    print(target_dir)
    classifier = ResNet18(BasicBlock=BasicBlock, nc=args.nc, nz=args.nz).to(device)
    classfier_path='../Speech_Command/Result/classifier/mel_spect_train_record/classifier.pth'
    str1 = 'epoch'
    str2 = 'best_cl_acc'
    try:
        checkpoint = torch.load(classfier_path, map_location={'cuda:1': cuda})
        classifier.load_state_dict(checkpoint['model'])
        best_epoch = checkpoint[str1]
        best_acc = checkpoint[str2]
        print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.8f})".format( classfier_path, best_epoch, best_acc))
    except:
        print("=> load classifier checkpoint '{}' failed".format(classfier_path))
        return

    #均设置采用评估模式
    classifier.eval()

    #记录绝对意义下的准确率和相对意义下的准确率
    no_noise_correct=0
    noise_correct=0
    relative_correct=0

    with torch.no_grad(): #禁用梯度计算
        for data,target in data_loader:
            data, target = data.to(device), target.to(device)
            #原始数据 分类结果
            prediction1 = classifier(data)
            pred1=prediction1.max(1,keepdim=True)[1]   #在指定维度计算张量的（最大值，对应索引）

            prediction2 = classifier(data,0,args.split_point,args.noise)
            pred2 = prediction2.max(1, keepdim=True)[1]

            no_noise_correct+=pred1.eq(target.view_as(pred1)).sum().item()
            noise_correct += pred2.eq(target.view_as(pred2)).sum().item()
            relative_correct += pred2.eq(pred1.view_as(pred2)).sum().item()

    print('no_noise_correct:',no_noise_correct,no_noise_correct/len(data_loader))
    print('noise_correct:',noise_correct,noise_correct/len(data_loader))
    print('relative_correct:',relative_correct,relative_correct/len(data_loader))
    with open(target_dir,'a') as file:
        file.write('\nno_noise_corrcect: {}/{}, {:8f}\n '.format(no_noise_correct,len(data_loader),no_noise_correct/len(data_loader)))
        file.write('\nnoise_corrcect: {}/{}, {:8f}\n '.format(noise_correct, len(data_loader),noise_correct / len(data_loader)))
        file.write('\nrelative_corrcect: {}/{}, {:8f}\n '.format(relative_correct, len(data_loader),relative_correct / len(data_loader)))


def main():
    args = parser.parse_args()
    print("================================")
    print(args)
    print("================================")

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

    # mel_sprect
    mel_spect_acc_100_set = StudioSet1(root="../Speech_Command/audio/mel_spect_acc_100", transform=transform)
    mel_spect_acc_all_set = StudioSet1(root="../Speech_Command/audio/test_set_mel_spect_numpy",transform=transform)
    mel_spect_acc_100_loader = torch.utils.data.DataLoader(mel_spect_acc_100_set, batch_size=1, shuffle=True, **kwargs)
    mel_spect_acc_all_loader = torch.utils.data.DataLoader(mel_spect_acc_all_set, batch_size=1, shuffle=True, **kwargs)

    os.makedirs('../Speech_Command/Result/noise_mel_spect_inversion/classifier_acc/', exist_ok=True)
    split=[1,3]
    noise=[1,2,3,4,5,6,7,8,9,10]
    noise2=[20,50,80]
    # 计算准确率acc
    for j in range(11, 301):
        for i in range(0,2):
            args.split_point=split[i]
            # args.noise=noise[j]
            # args.noise = noise2[j]
            args.noise=j

            mel_spect_acc_100_target_dir='../Speech_Command/Result/noise_mel_spect_inversion/classifier_acc/split' + str(args.split_point) + '_noise'+str(args.noise)+'_'+'acc_100.txt'
            mel_spect_acc_all_target_dir='../Speech_Command/Result/noise_mel_spect_inversion/classifier_acc/split' + str(args.split_point) + '_noise'+str(args.noise)+'_'+'acc_all.txt'
            # print(mel_spect_acc_all_target_dir)
            # print(mel_spect_acc_100_target_dir)

            classifier_acc(mel_spect_acc_100_loader,args,device, cuda,mel_spect_acc_100_target_dir)
            classifier_acc(mel_spect_acc_all_loader, args, device, cuda,mel_spect_acc_all_target_dir)

if __name__ == '__main__':
    main()