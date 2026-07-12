from __future__ import print_function
import argparse                      #用于解析命令行参数
import torch                         #用于构建和训练神经网络
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F      #用于图像预处理
from speech_command_dataset2 import StudioSet #数据集
# from speech_command_Model2 import BasicBlock,ResNet18,Conv_sp4, Conv_sp3, Conv_sp2, Conv_sp1  # 分类器，反转模型
from splitlearn_SC_Model import BasicBlock,ResNet18,Conv_sp4, Conv_sp3, Conv_sp2, Conv_sp1  # 分类器，反转模型
import numpy as np
import random
import os                    #用于文件和目录操作
from utils import TV
import time
import librosa  #用于音频处理和特征提取
import librosa.display  # 用于音频特征的可视化
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from evaluation_snr import SNR
from matplotlib.colors import LinearSegmentedColormap

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

#计算mfcc 100个音频的正确率
def mfcc_acc(data_loader,args,device,cuda,target_dir):
    print(target_dir)

    for j in [0, 5, 10, 15, 20, 25, 50, 100]:
        # 从指定的路径加载一个训练好的模型及其训练信息
        classifier = ResNet18(BasicBlock=BasicBlock, nc=args.nc, nz=args.nz).to(device)
        path = '../SplitLearn/Speech_Command/mel_spect/classifier/classifier_' + str(j) + '.pth'
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

        inversion_path='../SplitLearn/Speech_Command/mel_spect/whitebox_inversion/split_point'+str(args.split_point)+'/inversion_'+str(j)+'/train_record/inversion.pth'

        str1 = 'best_epoch'
        str2 = 'best_acc'
        try:
            checkpoint = torch.load(inversion_path, map_location={'cuda:'+str(args.cuda_id): cuda})
            inversion.load_state_dict(checkpoint['model'])
            best_epoch = checkpoint[str1]
            best_acc = checkpoint[str2]
            print("=> loaded inversion checkpoint '{}' (epoch {}, acc {:.8f})".format(inversion_path, best_epoch, best_acc))
        except:
            print("=> load inversion checkpoint '{}' failed".format(inversion_path))
            return

        #均设置采用评估模式
        classifier.eval()
        inversion.eval()

        #记录绝对意义下的准确率和相对意义下的准确率
        absolute_correct=0
        relative_correct=0
        origin_correct=0

        with torch.no_grad(): #禁用梯度计算
            for data,target in data_loader:
                data, target = data.to(device), target.to(device)
                #原始数据 分类结果
                prediction = classifier(data)
                pred1=prediction.max(1,keepdim=True)[1]   #在指定维度计算张量的（最大值，对应索引）

                #反转数据 分类结果
                split_data = classifier(data,split_point=args.split_point)
                reconstruction = inversion(split_data)
                reconstruction_prediction=classifier(reconstruction)
                pred2=reconstruction_prediction.max(1,keepdim=True)[1]

                origin_correct+=pred1.eq(target.view_as(pred1)).sum().item()
                absolute_correct += pred2.eq(target.view_as(pred2)).sum().item()
                relative_correct += pred2.eq(pred1.view_as(pred2)).sum().item()

        print('origin_correct:',origin_correct,origin_correct/len(data_loader))
        print('absolute_correct:',absolute_correct,absolute_correct/len(data_loader))
        print('relative_correct:',relative_correct,relative_correct/len(data_loader))

        os.makedirs(target_dir+'split_point'+str(args.split_point)+'/', exist_ok=True)
        now_dir=target_dir+'split_point'+str(args.split_point)+'/inversion'+str(j)+'.txt'
        print(now_dir)
        with open(now_dir,'a') as file:
            file.write('\norigin_corrcect: {}/{}, {:8f}\n '.format(origin_correct,len(data_loader),origin_correct/len(data_loader)))
            file.write('\nabsolute_corrcect: {}/{}, {:8f}\n '.format(absolute_correct, len(data_loader),absolute_correct / len(data_loader)))
            file.write('\nrelative_corrcect: {}/{}, {:8f}\n '.format(relative_correct, len(data_loader),relative_correct / len(data_loader)))

def mfcc_mse_plot(data_loader, args, device, cuda, target_dir):
    print(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # 初始化一个空列表，用于存储所有数据
    all_data = []

    # 遍历不同的 inversion 模型
    for j in [0, 5, 10, 15, 20, 25, 50, 100]:
    # for j in [0, 5, 10, 20,100]:
        # 从指定的路径加载一个训练好的模型及其训练信息
        classifier = ResNet18(BasicBlock=BasicBlock, nc=args.nc, nz=args.nz).to(device)
        path = '../SplitLearn/Speech_Command/mfcc/classifier/classifier_' + str(j) + '.pth'
        print(path)
        try:
            checkpoint = torch.load(path, map_location={'cuda:' + str(args.cuda_id): cuda})
            classifier.load_state_dict(checkpoint['model'])
            best_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.8f})".format(path, best_epoch, best_acc))
        except:
            print("=> load classifier checkpoint '{}' failed".format(path))
            return

        # 遍历不同的分割点
        for i in range(4):  # 代表四个分割点
            # 根据分割点创建 inversion 模型
            if i == 0:
                inversion = Conv_sp1().to(device)
            elif i == 1:
                inversion = Conv_sp2().to(device)
            elif i == 2:
                inversion = Conv_sp3().to(device)
            elif i == 3:
                inversion = Conv_sp4().to(device)

            inversion_path = '../SplitLearn/Speech_Command/mfcc/blackbox_inversion/split_point' + str(i + 1) + '/inversion_' + str(j) + '/train_record/inversion.pth'
            try:
                checkpoint = torch.load(inversion_path, map_location={'cuda:' + str(args.cuda_id): cuda})
                inversion.load_state_dict(checkpoint['model'])
                best_epoch = checkpoint['best_epoch']
                best_acc = checkpoint['best_acc']
                print("=> loaded inversion checkpoint '{}' (epoch {}, acc {:.8f})".format(inversion_path, best_epoch, best_acc))
            except:
                print("=> load inversion checkpoint '{}' failed".format(inversion_path))
                return

            classifier.eval()
            inversion.eval()

            # 用于存储当前分割点和 inversion 模型的 MSE 值
            layer_record_mse = []

            with torch.no_grad():  # 禁用梯度计算
                for data, target in data_loader:
                    data, target = data.to(device), target.to(device)
                    split_data = classifier(data, i + 1)
                    reconstruction = inversion(split_data)

                    temp_mse_loss = F.mse_loss(reconstruction, data)
                    layer_record_mse.append(temp_mse_loss.item())

            # 将当前分割点和 inversion 模型的 MSE 值添加到 all_data 列表中
            for mse in layer_record_mse:
                all_data.append({
                    'split': f'Split{i + 1}',
                    'inversion': j,
                    'mse': mse
                })

    # # 将 all_data 转换为 DataFrame
    # df = pd.DataFrame(all_data)
    #
    # # 保存 all_data 为 CSV 文件
    # csv_path = os.path.join(target_dir, 'all_data.csv')
    # df.to_csv(csv_path, index=False)
    # print(f"Data saved to {csv_path}")
    #
    # # 提取每个 split 下的数据
    # split1_mse = [df[(df['split'] == 'Split1') & (df['inversion'] == j)]['mse'].tolist() for j in [0, 5, 10, 20, 100]]
    # split2_mse = [df[(df['split'] == 'Split2') & (df['inversion'] == j)]['mse'].tolist() for j in [0, 5, 10, 20, 100]]
    # split3_mse = [df[(df['split'] == 'Split3') & (df['inversion'] == j)]['mse'].tolist() for j in [0, 5, 10, 20, 100]]
    # split4_mse = [df[(df['split'] == 'Split4') & (df['inversion'] == j)]['mse'].tolist() for j in [0, 5, 10, 20, 100]]
    #
    # # 定义颜色映射
    # colors = [
    #     (150 / 255., 165 / 255., 210 / 255.),  # 稍微深一点的蓝色
    #     (245 / 255., 145 / 255., 130 / 255.)  # 稍微深一点的橙色
    # ]
    # cmap = LinearSegmentedColormap.from_list("custom", colors, N=5)  # 生成 5 种颜色
    # # 提取颜色
    # cmap_colors = [cmap(i) for i in range(cmap.N)]
    #
    # # 定义标签
    # labels = ['$0$', '$5$', '$10$', '$20$', '$100$']
    #
    # # 绘制箱线图
    # plt.figure(figsize=(10, 6))  # 调整图像大小
    #
    # # Split1
    # bplot1 = plt.boxplot(split1_mse, patch_artist=True, labels=labels, positions=[1, 1.4, 1.8, 2.2, 2.6], widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    # for patch, color in zip(bplot1['boxes'], cmap_colors):
    #     patch.set_facecolor(color)
    # for median in bplot1['medians']:
    #     median.set(color='black')
    #
    # # Split2
    # bplot2 = plt.boxplot(split2_mse, patch_artist=True, labels=labels, positions=[3.5, 3.9, 4.3, 4.7, 5.1], widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    # for patch, color in zip(bplot2['boxes'], cmap_colors):
    #     patch.set_facecolor(color)
    # for median in bplot2['medians']:
    #     median.set(color='black')
    #
    # # Split3
    # bplot3 = plt.boxplot(split3_mse, patch_artist=True, labels=labels, positions=[6, 6.4, 6.8, 7.2, 7.6], widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    # for patch, color in zip(bplot3['boxes'], cmap_colors):
    #     patch.set_facecolor(color)
    # for median in bplot3['medians']:
    #     median.set(color='black')
    #
    # # Split4
    # bplot4 = plt.boxplot(split4_mse, patch_artist=True, labels=labels, positions=[8.5, 8.9, 9.3, 9.7, 10.1], widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    # for patch, color in zip(bplot4['boxes'], cmap_colors):
    #     patch.set_facecolor(color)
    # for median in bplot4['medians']:
    #     median.set(color='black')
    #
    # # 设置 x 轴的刻度标签
    # x_position = [1.8, 4.3, 6.8, 9.3]  # 每组箱型图的中间位置
    # x_position_fmt = ["$Split1$", "$Split2$", "$Split3$", "$Split4$"]
    # plt.xticks(x_position, x_position_fmt, fontsize=16)
    #
    # # 设置 y 轴标签
    # plt.yticks(fontsize=16)
    # plt.ylabel('MSE', fontsize=18)
    #
    # # 添加网格线
    # plt.grid(linestyle="--", alpha=0.3)
    #
    # # 设置图例，并将其放在左上方
    # plt.legend(bplot1['boxes'], labels, loc='upper left', title='Epoch', fontsize=12, title_fontsize=14)
    #
    # # # 设置 y 轴范围
    # # plt.ylim([0, 0.035])
    #
    # # 调整布局
    # plt.tight_layout()
    #
    # # 保存图表
    # plt.savefig(os.path.join(target_dir, 'mse.png'), dpi=800, bbox_inches='tight')
    #
    # # 显示图表
    # plt.show()

    df = pd.DataFrame(all_data) #将字典data转换为pandas数据框df
    # 绘制箱线图
    plt.figure(figsize=(12, 6))  # 调整图像大小
    sns.boxplot(data=df, x='split', y='mse', hue='inversion', palette="coolwarm", showmeans=True)
    plt.xticks(fontsize=18)  # 设置字体
    plt.yticks(fontsize=18)
    plt.xlabel("Split Point", fontsize=24)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=24)
    plt.title("MSE for Different Inversions and Split Points", fontsize=24)
    plt.legend(title="Inversion", fontsize=16, title_fontsize=16)
    plt.grid(axis="y", linestyle="--")  # 添加水平网格线
    plt.tight_layout()
    plt.show()

def mfcc_snr_plot(data_loader, args, device, cuda, target_dir):
    print(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # 初始化一个空列表，用于存储所有数据
    all_data = []

    # 遍历不同的 inversion 模型
    # for j in [0, 5, 10, 15, 20, 25, 50, 100]:
    for j in [0, 5, 10, 20, 100]:
        # 从指定的路径加载一个训练好的模型及其训练信息
        classifier = ResNet18(BasicBlock=BasicBlock, nc=args.nc, nz=args.nz).to(device)
        path = '../SplitLearn/Speech_Command/mfcc/classifier/classifier_' + str(j) + '.pth'
        print(path)
        try:
            checkpoint = torch.load(path, map_location={'cuda:' + str(args.cuda_id): cuda})
            classifier.load_state_dict(checkpoint['model'])
            best_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
            print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.8f})".format(path, best_epoch, best_acc))
        except:
            print("=> load classifier checkpoint '{}' failed".format(path))
            return

        # 遍历不同的分割点
        for i in range(4):  # 代表四个分割点
            # 根据分割点创建 inversion 模型
            if i == 0:
                inversion = Conv_sp1().to(device)
            elif i == 1:
                inversion = Conv_sp2().to(device)
            elif i == 2:
                inversion = Conv_sp3().to(device)
            elif i == 3:
                inversion = Conv_sp4().to(device)

            inversion_path = '../SplitLearn/Speech_Command/mfcc/blackbox_inversion/split_point' + str(i + 1) + '/inversion_' + str(j) + '/train_record/inversion.pth'
            try:
                checkpoint = torch.load(inversion_path, map_location={'cuda:' + str(args.cuda_id): cuda})
                inversion.load_state_dict(checkpoint['model'])
                best_epoch = checkpoint['best_epoch']
                best_acc = checkpoint['best_acc']
                print("=> loaded inversion checkpoint '{}' (epoch {}, acc {:.8f})".format(inversion_path, best_epoch, best_acc))
            except:
                print("=> load inversion checkpoint '{}' failed".format(inversion_path))
                return

            classifier.eval()
            inversion.eval()

            # 用于存储当前分割点和 inversion 模型的 SNR 值
            layer_record_snr = []

            with torch.no_grad():  # 禁用梯度计算
                for data, target in data_loader:
                    data, target = data.to(device), target.to(device)
                    split_data = classifier(data, i + 1)
                    reconstruction = inversion(split_data)

                    # 将 data 和 reconstruction 转换为 NumPy 数组
                    data_numpy = data.cpu().detach().numpy()
                    reconstruction_numpy = reconstruction.cpu().detach().numpy()

                    # 计算 SNR
                    snr = SNR(data_numpy, data_numpy - reconstruction_numpy)
                    layer_record_snr.append(snr)

            # 将当前分割点和 inversion 模型的 SNR 值添加到 all_data 列表中
            for snr in layer_record_snr:
                all_data.append({
                    'split': f'Split{i + 1}',
                    'inversion': j,
                    'snr': snr
                })

    # 将 all_data 转换为 DataFrame
    df = pd.DataFrame(all_data)

    # 保存 all_data 为 CSV 文件
    csv_path = os.path.join(target_dir, 'all_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

    # 提取每个 split 下的数据
    split1_snr = [df[(df['split'] == 'Split1') & (df['inversion'] == j)]['snr'].tolist() for j in [0, 5, 10, 20, 100]]
    split2_snr = [df[(df['split'] == 'Split2') & (df['inversion'] == j)]['snr'].tolist() for j in [0, 5, 10, 20, 100]]
    split3_snr = [df[(df['split'] == 'Split3') & (df['inversion'] == j)]['snr'].tolist() for j in [0, 5, 10, 20, 100]]
    split4_snr = [df[(df['split'] == 'Split4') & (df['inversion'] == j)]['snr'].tolist() for j in [0, 5, 10, 20, 100]]

    # 定义颜色映射
    colors = [
        (150 / 255., 165 / 255., 210 / 255.),  # 稍微深一点的蓝色
        (245 / 255., 145 / 255., 130 / 255.)  # 稍微深一点的橙色
    ]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=5)  # 生成 5 种颜色
    # 提取颜色
    cmap_colors = [cmap(i) for i in range(cmap.N)]

    # 定义标签
    labels = ['$0$', '$5$', '$10$', '$20$', '$100$']

    # 绘制箱线图
    plt.figure(figsize=(10, 6))  # 调整图像大小

    # Split1
    bplot1 = plt.boxplot(split1_snr, patch_artist=True, labels=labels, positions=[1, 1.4, 1.8, 2.2, 2.6], widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot1['boxes'], cmap_colors):
        patch.set_facecolor(color)
    for median in bplot1['medians']:
        median.set(color='black')

    # Split2
    bplot2 = plt.boxplot(split2_snr, patch_artist=True, labels=labels, positions=[3.5, 3.9, 4.3, 4.7, 5.1], widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], cmap_colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')

    # Split3
    bplot3 = plt.boxplot(split3_snr, patch_artist=True, labels=labels, positions=[6, 6.4, 6.8, 7.2, 7.6], widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], cmap_colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')

    # Split4
    bplot4 = plt.boxplot(split4_snr, patch_artist=True, labels=labels, positions=[8.5, 8.9, 9.3, 9.7, 10.1], widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], cmap_colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')

    # 设置 x 轴的刻度标签
    x_position = [1.8, 4.3, 6.8, 9.3]  # 每组箱型图的中间位置
    x_position_fmt = ["$Split1$", "$Split2$", "$Split3$", "$Split4$"]
    plt.xticks(x_position, x_position_fmt, fontsize=16)

    # 设置 y 轴标签
    plt.yticks(fontsize=16)
    plt.ylabel('SNR', fontsize=18)

    # 添加网格线
    plt.grid(linestyle="--", alpha=0.3)

    # 设置图例，并将其放在左上方
    plt.legend(bplot1['boxes'], labels, loc='upper right', title='Epoch', fontsize=12, title_fontsize=14)

    # #设置y轴范围
    # plt.ylim([0, 50])

    # 调整布局
    plt.tight_layout()

    # 保存图表
    plt.savefig(os.path.join(target_dir, 'snr.png'), dpi=800, bbox_inches='tight')

    # 显示图表
    plt.show()


    # df = pd.DataFrame(all_data) #将字典data转换为pandas数据框df
    # # 绘制箱线图
    # plt.figure(figsize=(12, 6))  # 调整图像大小
    # sns.boxplot(data=df, x='split', y='snr', hue='inversion', palette="coolwarm", showmeans=True)
    # plt.xticks(fontsize=18)  # 设置字体
    # plt.yticks(fontsize=18)
    # plt.xlabel("Split Point", fontsize=24)
    # plt.ylabel("Signal-to-Noise Ratio (SNR)", fontsize=24)
    # plt.title("SNR for Different Inversions and Split Points", fontsize=24)
    # plt.legend(title="Inversion", fontsize=16, title_fontsize=16)
    # plt.grid(axis="y", linestyle="--")  # 添加水平网格线
    # plt.tight_layout()
    # plt.show()

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

    # mfcc
    mfcc_acc_100_set = StudioSet(root="../Speech_Command/audio/mfcc_acc_100", transform=transform)
    mfcc_acc_100_loader = torch.utils.data.DataLoader(mfcc_acc_100_set, batch_size=1, shuffle=True, **kwargs)

    # # 计算准确率acc
    # for i in range(1,5):
    #     args.split_point=i
    #     # target_dir_root = '../SplitLearn/Speech_Command/Result/mel_whitebox/acc/'
    #     # mel_spect_acc_100_target_dir='../Speech_Command/Result/mel_spect_inversion/split_point'+str(i)+'/train_record/acc_100.txt'
    #     mel_spect_acc_100_target_dir='../SplitLearn/Speech_Command/Result/acc/mel_whitebox/'
    #     mel_spect_acc(mel_spect_acc_100_loader,args,device, cuda,mel_spect_acc_100_target_dir)

    # # 绘制mse图像
    mfcc_mse_100_target_dir ='../SplitLearn/Speech_Command/Result/mse/mfcc_blackbox/'
    mfcc_mse_plot(mfcc_acc_100_loader,args,device,cuda,mfcc_mse_100_target_dir)

    # # # 绘制snr图像
    # mfcc_snr_100_target_dir = '../SplitLearn/Speech_Command/Result/snr/mfcc_blackbox/'
    # mfcc_snr_plot(mfcc_acc_100_loader,args,device,cuda,mfcc_snr_100_target_dir)

if __name__ == '__main__':
    main()

