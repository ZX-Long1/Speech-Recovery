from __future__ import print_function
import argparse                      #用于解析命令行参数
import torch                         #用于构建和训练神经网络
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F      #用于图像预处理
from speech_command_dataset2 import StudioSet as StudioSet #数据集
from speech_command_Model2 import BasicBlock,ResNet18,Conv_sp4, Conv_sp3, Conv_sp2, Conv_sp1  # 分类器，反转模型
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from evaluation_snr import SNR

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

def mfcc_SNR(data_loader,args,device,cuda,target_dir):
    print(target_dir)
    classifier = ResNet18(BasicBlock=BasicBlock, nc=args.nc, nz=args.nz).to(device)
    classfier_path = '../Speech_Command/Result/classifier/mfcc_train_record/classifier.pth'
    str1 = 'epoch'
    str2 = 'best_cl_acc'
    try:
        checkpoint = torch.load(classfier_path, map_location={'cuda:1': cuda})
        classifier.load_state_dict(checkpoint['model'])
        best_epoch = checkpoint[str1]
        best_acc = checkpoint[str2]
        print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.8f})".format(classfier_path, best_epoch, best_acc))
    except:
        print("=> load classifier checkpoint '{}' failed".format(classfier_path))
        return

    if args.split_point == 1:
        inversion = Conv_sp1().to(device)
    elif args.split_point == 2:
        inversion = Conv_sp2().to(device)
    elif args.split_point == 3:
        inversion = Conv_sp3().to(device)
    elif args.split_point == 4:
        inversion = Conv_sp4().to(device)

    inversion_path = '../Speech_Command/Result/mfcc_inversion/split_point' + str(args.split_point) + '/train_record/inversion.pth'

    str1 = 'best_epoch'
    str2 = 'best_acc'
    try:
        checkpoint = torch.load(inversion_path, map_location={'cuda:1': cuda})
        inversion.load_state_dict(checkpoint['model'])
        best_epoch = checkpoint[str1]
        best_acc = checkpoint[str2]
        print("=> loaded inversion checkpoint '{}' (epoch {}, acc {:.8f})".format(inversion_path, best_epoch, best_acc))
    except:
        print("=> load inversion checkpoint '{}' failed".format(inversion_path))
        return

    # 均设置采用评估模式
    classifier.eval()
    inversion.eval()

    # 记录绝对意义下的准确率和相对意义下的准确率
    snr = 0

    with torch.no_grad():  # 禁用梯度计算
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            split_data = classifier(data, split_point=args.split_point)
            reconstruction = inversion(split_data)
            # snr += SNR(data, reconstruction)

            # 将 data 和 reconstruction 转换为 NumPy 数组
            data_numpy = data.cpu().detach().numpy()
            reconstruction_numpy = reconstruction.cpu().detach().numpy()
            # 调用 SNR 函数，传入 NumPy 数组
            snr += SNR(data_numpy, data_numpy-reconstruction_numpy)

        snr /= len(data_loader)

    print('SNR:',snr)

    with open(target_dir, 'a') as file:
        file.write('\nSNR: {:8f}\n'.format(snr))

def mfcc_acc(data_loader,args,device,cuda,target_dir):
    print(target_dir)
    classifier = ResNet18(BasicBlock=BasicBlock, nc=args.nc, nz=args.nz).to(device)
    classfier_path = '../Speech_Command/Result/classifier/mfcc_train_record/classifier.pth'
    str1 = 'epoch'
    str2 = 'best_cl_acc'
    try:
        checkpoint = torch.load(classfier_path, map_location={'cuda:1': cuda})
        classifier.load_state_dict(checkpoint['model'])
        best_epoch = checkpoint[str1]
        best_acc = checkpoint[str2]
        print(
            "=> loaded classifier checkpoint '{}' (epoch {}, acc {:.8f})".format(classfier_path, best_epoch, best_acc))
    except:
        print("=> load classifier checkpoint '{}' failed".format(classfier_path))
        return

    if args.split_point == 1:
        inversion = Conv_sp1().to(device)
    elif args.split_point == 2:
        inversion = Conv_sp2().to(device)
    elif args.split_point == 3:
        inversion = Conv_sp3().to(device)
    elif args.split_point == 4:
        inversion = Conv_sp4().to(device)

    inversion_path = '../Speech_Command/Result/blackbox_mfcc_inversion/split_point' + str(args.split_point) + '/train_record/inversion.pth'

    str1 = 'best_epoch'
    str2 = 'best_acc'
    try:
        checkpoint = torch.load(inversion_path, map_location={'cuda:1': cuda})
        inversion.load_state_dict(checkpoint['model'])
        best_epoch = checkpoint[str1]
        best_acc = checkpoint[str2]
        print("=> loaded inversion checkpoint '{}' (epoch {}, acc {:.8f})".format(inversion_path, best_epoch, best_acc))
    except:
        print("=> load inversion checkpoint '{}' failed".format(inversion_path))
        return

    # 均设置采用评估模式
    classifier.eval()
    inversion.eval()

    # 记录绝对意义下的准确率和相对意义下的准确率
    absolute_correct = 0
    relative_correct = 0
    origin_correct = 0

    with torch.no_grad():  # 禁用梯度计算
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            # 原始数据 分类结果
            prediction = classifier(data)
            pred1 = prediction.max(1, keepdim=True)[1]  # 在指定维度计算张量的（最大值，对应索引）

            # 反转数据 分类结果
            split_data = classifier(data, split_point=args.split_point)
            reconstruction = inversion(split_data)
            reconstruction_prediction = classifier(reconstruction)
            pred2 = reconstruction_prediction.max(1, keepdim=True)[1]

            origin_correct += pred1.eq(target.view_as(pred1)).sum().item()
            absolute_correct += pred2.eq(target.view_as(pred2)).sum().item()
            relative_correct += pred2.eq(pred1.view_as(pred2)).sum().item()

    print('origin_correct:', origin_correct, origin_correct / len(data_loader))
    print('absolute_correct:', absolute_correct, absolute_correct / len(data_loader))
    print('relative_correct:', relative_correct, relative_correct / len(data_loader))
    with open(target_dir, 'a') as file:
        file.write('\norigin_corrcect: {}/{}, {:8f}\n '.format(origin_correct, len(data_loader),origin_correct / len(data_loader)))
        file.write('\nabsolute_corrcect: {}/{}, {:8f}\n '.format(absolute_correct, len(data_loader),absolute_correct / len(data_loader)))
        file.write('\nrelative_corrcect: {}/{}, {:8f}\n '.format(relative_correct, len(data_loader),relative_correct / len(data_loader)))


def mfcc_mse_plot(data_loader,args,device,cuda,target_dir):
    # print(target_dir)
    # os.makedirs(target_dir, exist_ok=True)
    classifier = ResNet18(BasicBlock=BasicBlock, nc=args.nc, nz=args.nz).to(device)
    classfier_path = '../Speech_Command/Result/classifier/mfcc_train_record/classifier.pth'
    str1 = 'epoch'
    str2 = 'best_cl_acc'
    try:
        checkpoint = torch.load(classfier_path, map_location={'cuda:1': cuda})
        classifier.load_state_dict(checkpoint['model'])
        best_epoch = checkpoint[str1]
        best_acc = checkpoint[str2]
        print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.8f})".format(classfier_path, best_epoch, best_acc))
    except:
        print("=> load classifier checkpoint '{}' failed".format(classfier_path))
        return

    record_mse = []
    mode = "BlackBox"  # 可改 WhiteBox BlackBox
    for i in range(4): #代表四个分割点
        layer_record_mse = []

        #根据分割点创建inversion模型
        if i == 0:
            inversion = Conv_sp1().to(device)
        elif i == 1:
            inversion = Conv_sp2().to(device)
        elif i == 2:
            inversion = Conv_sp3().to(device)
        elif i == 3:
            inversion = Conv_sp4().to(device)

        inversion_path = '../Speech_Command/Result/blackbox_mfcc_inversion/split_point' + str(i+1) + '/train_record/inversion.pth'

        str1 = 'best_epoch'
        str2 = 'best_acc'
        try:
            checkpoint = torch.load(inversion_path, map_location={'cuda:1': cuda})
            inversion.load_state_dict(checkpoint['model'])
            best_epoch = checkpoint[str1]
            best_acc = checkpoint[str2]
            print("=> loaded inversion checkpoint '{}' (epoch {}, acc {:.8f})".format(inversion_path, best_epoch,best_acc))
        except:
            print("=> load inversion checkpoint '{}' failed".format(inversion_path))
            return

        classifier.eval()
        inversion.eval()

    #     with torch.no_grad():  # 禁用梯度计算
    #         for data, target in data_loader:
    #             data, target = data.to(device), target.to(device)
    #             split_data = classifier(data, i+1)
    #             reconstruction = inversion(split_data)
    #
    #             temp_mse_loss=F.mse_loss(reconstruction, data)
    #             layer_record_mse.append(temp_mse_loss.item())
    #         record_mse.append(layer_record_mse)
    #
    #     # 保存mse数组
    # os.makedirs('../mse_plot/', exist_ok=True)
    # mse_dir = '../mse_plot/SpeechCommand_blackbox_mfcc.npy'
    # np.save(mse_dir, record_mse)
        # 预处理snr数组
        with torch.no_grad():  # 禁用梯度计算
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                split_data = classifier(data, i + 1)
                reconstruction = inversion(split_data)
                data_numpy = data.cpu().detach().numpy()
                reconstruction_numpy = reconstruction.cpu().detach().numpy()
                # 调用 SNR 函数，传入 NumPy 数组

                snr = SNR(data_numpy, data_numpy - reconstruction_numpy)
                layer_record_mse.append(snr.item())
            record_mse.append(layer_record_mse)

    # #保存snr数组
    os.makedirs('../snr_plot/', exist_ok=True)
    mse_dir = '../snr_plot/SpeechCommand_blackbox_mfcc.npy'
    np.save(mse_dir, record_mse)

    # data = {} #空字典，存放每个分组的均方误差（采用键值对的形式）
    # layers = ["split1", "split2", "split3", "split4"]
    # for i in range(0, 4):
    #     data[layers[i]] = record_mse[i]
    #
    #     df = pd.DataFrame(data)  # 将字典data转换为pandas数据框df
    #     # 创建绘图
    #     plt.figure(figsize=(8, 6))  # 调整图像大小
    #     # 绘制箱线图
    #     plt.boxplot(df,
    #                 patch_artist=True,
    #                 boxprops=dict(linewidth=1.5),
    #                 whiskerprops=dict(linewidth=1.5),
    #                 capprops=dict(linewidth=1.5),
    #                 medianprops=dict(linewidth=2, color='red'),
    #                 meanprops=dict(color='black'))
    #     # 获取当前的Axes对象
    #     ax = plt.gca()
    #     # 设置背景色
    #     ax.set_facecolor("white")
    #     # 显示网格
    #     ax.grid(True, which='both', axis='y', linestyle='--', color='black', linewidth=1)
    #     # 设置X轴标签
    #     ax.set_xticklabels(layers)
    #     # 调整坐标轴标签
    #     plt.xticks(fontsize=18)  # 设置字体
    #     plt.yticks(fontsize=18)
    #     plt.ylabel("Mean Squared Error (MSE)", fontsize=24)
    #     # 设置 y 轴范围和网格
    #     plt.ylim([0, 0.01])  # 设置 y 轴范围
    #     # 保存图像
    #     plt.tight_layout()
    #     plt.savefig(target_dir + '1.png', dpi=600, bbox_inches="tight")
    #     plt.show()
    #
    #     df = pd.DataFrame(data)  # 将字典data转换为pandas数据框df
    #     plt.figure(figsize=(8, 6))  # 调整图像大小
    #     sns.boxplot(data=df, width=0.6, palette="coolwarm", showmeans=True)  # coolwarm 渐变配色，添加均值线
    #     # 获取当前的Axes对象
    #     ax = plt.gca()
    #     ax.set_xticklabels(layers)
    #     plt.xticks(fontsize=18)  # 设置字体
    #     plt.yticks(fontsize=18)
    #     plt.ylabel("Mean Squared Error (MSE)", fontsize=24)
    #     plt.ylim([0, 0.01])  # 设置 y 轴范围
    #     plt.grid(axis="y", linestyle="--")  # 添加水平网格线
    #     plt.tight_layout()
    #     plt.savefig(target_dir + '2.png', dpi=600, bbox_inches="tight")
    #     plt.show()

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

    #mfcc
    mfcc_acc_100_set = StudioSet(root="../Speech_Command/audio/mfcc_acc_100", transform=transform)
    mfcc_acc_all_set = StudioSet(root="../Speech_Command/audio/test_set_mfcc_numpy",transform=transform)
    mfcc_acc_100_loader = torch.utils.data.DataLoader(mfcc_acc_100_set, batch_size=1, shuffle=True, **kwargs)
    mfcc_acc_all_loader = torch.utils.data.DataLoader(mfcc_acc_all_set, batch_size=1, shuffle=True, **kwargs)

    # # 计算准确率acc
    # for i in range(1,5):
    #     args.split_point=i
    #     mfcc_acc_100_target_dir='../Speech_Command/Result/blackbox_mfcc_inversion/split_point'+str(args.split_point)+'/train_record/acc_100.txt'
    #     mfcc_acc_all_target_dir='../Speech_Command/Result/blackbox_mfcc_inversion/split_point'+str(args.split_point)+'/train_record/acc_all.txt'
    #     mfcc_acc(mfcc_acc_100_loader,args,device, cuda,mfcc_acc_100_target_dir)
    #     mfcc_acc(mfcc_acc_all_loader, args, device, cuda,mfcc_acc_all_target_dir)
    #
    # # 绘制mse图像
    # mfcc_mse_100_target_dir='../Speech_Command/Result/blackbox_mfcc_inversion/mfcc_mse/mfcc_mse_100_'
    # mfcc_mse_all_target_dir='../Speech_Command/Result/blackbox_mfcc_inversion/mfcc_mse/mfcc_mse_all_'
    # mfcc_mse_plot(mfcc_acc_100_loader,args,device,cuda,mfcc_mse_100_target_dir)
    # mfcc_mse_plot(mfcc_acc_all_loader,args, device, cuda, mfcc_mse_all_target_dir)

    # os.makedirs('../SNR/SpeechCommand/', exist_ok=True)
    # for i in range(1, 5):
    #     args.split_point = i
    #     mel_spect_acc_100_target_dir = '../SNR/SpeechCommand/mfcc_blackbox_split' + str(i) + '_100.txt'
    #     mel_spect_acc_all_target_dir = '../SNR/SpeechCommand/mfcc_blackbox_split' + str(i) + '_all.txt'
    #
    #     mfcc_SNR(mfcc_acc_100_loader, args, device, cuda, mel_spect_acc_100_target_dir)
    #     mfcc_SNR(mfcc_acc_all_loader, args, device, cuda, mel_spect_acc_all_target_dir)

    mfcc_mse_plot(mfcc_acc_100_loader, args, device, cuda, '')


if __name__ == '__main__':
    main()