from __future__ import print_function
import argparse  # 用于解析命令行参数
import torch  # 用于构建和训练神经网络
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F  # 用于图像预处理
from speech_command_dataset1 import StudioSet  # 数据集
from speech_command_Model import BasicBlock, ResNet18, Conv_sp4, Conv_sp3, Conv_sp2, Conv_sp1  # 分类器，反转模型
import numpy as np
import random
import os  # 用于文件和目录操作
from utils import TV
import time
import librosa  # 用于音频处理和特征提取
import librosa.display  # 用于音频特征的可视化
import matplotlib.pyplot as plt
from evaluation_snr import SNR

parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='')
parser.add_argument('--epochs', type=int, default=100, metavar='')
parser.add_argument('--lr', type=float, default=0.0002, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log_interval', type=int, default=50, metavar='')
parser.add_argument('--nc', type=int, default=1)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nz', type=int, default=35)
parser.add_argument('--num_workers', type=int, default=50, metavar='')
parser.add_argument('--split_point', type=int, default=1, metavar='')
parser.add_argument('--cuda_id', type=int, default=2, metavar='')
parser.add_argument('--nTV', type=float, default=1, metavar='')
parser.add_argument('--dropout', type=int, default=0, metavar='')

def  dropout_acc(data_loader,args,device,cuda,target_dir):
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

    if args.split_point == 1:
        inversion = Conv_sp1().to(device)
    elif args.split_point == 2:
        inversion = Conv_sp2().to(device)
    elif args.split_point == 3:
        inversion = Conv_sp3().to(device)
    elif args.split_point == 4:
        inversion = Conv_sp4().to(device)

    inversion_path = '../Speech_Command/Result/dropout_mel_spect_inversion/split' + str(args.split_point) +'_dropout'+str(args.dropout) +'/train_record/inversion.pth'

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
    with open(target_dir,'a') as file:
        file.write('\norigin_corrcect: {}/{}, {:8f}\n '.format(origin_correct,len(data_loader),origin_correct/len(data_loader)))
        file.write('\nabsolute_corrcect: {}/{}, {:8f}\n '.format(absolute_correct, len(data_loader),absolute_correct / len(data_loader)))
        file.write('\nrelative_corrcect: {}/{}, {:8f}\n '.format(relative_correct, len(data_loader),relative_correct / len(data_loader)))

def dropout_SNR(data_loader,args,device,cuda,target_dir):
    print(target_dir)
    classifier = ResNet18(BasicBlock=BasicBlock, nc=args.nc, nz=args.nz).to(device)
    classfier_path = '../Speech_Command/Result/classifier/mel_spect_train_record/classifier.pth'
    str1 = 'epoch'
    str2 = 'best_cl_acc'
    try:
        checkpoint = torch.load(classfier_path, map_location={'cuda:1': cuda})
        classifier.load_state_dict(checkpoint['model'])
        best_epoch = checkpoint[str1]
        best_acc = checkpoint[str2]
        # print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.8f})".format(classfier_path, best_epoch, best_acc))
    except:
        # print("=> load classifier checkpoint '{}' failed".format(classfier_path))
        return

    if args.split_point == 1:
        inversion = Conv_sp1().to(device)
    elif args.split_point == 2:
        inversion = Conv_sp2().to(device)
    elif args.split_point == 3:
        inversion = Conv_sp3().to(device)
    elif args.split_point == 4:
        inversion = Conv_sp4().to(device)

    inversion_path = '../Speech_Command/Result/dropout_mel_spect_inversion/split' + str(args.split_point) +'_dropout'+str(args.dropout) +'/train_record/inversion.pth'

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
def dropout_mse_plot(data_loader,args,device,cuda,target_dir):
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

    record_mse = []
    mode = "WhiteBox"  # 可改 WhiteBox BlackBox
    dropout=[0,1,2,3,4,5]
    for i in range(6):  # 代表四个分割点
        args.dropout=dropout[i]
        layer_record_mse = []

        if args.split_point == 1:
            inversion = Conv_sp1().to(device)
        elif args.split_point == 2:
            inversion = Conv_sp2().to(device)
        elif args.split_point == 3:
            inversion = Conv_sp3().to(device)
        elif args.split_point == 4:
            inversion = Conv_sp4().to(device)

        inversion_path = '../Speech_Command/Result/dropout_mel_spect_inversion/split' + str(args.split_point) +'_dropout'+str(args.dropout) +'/train_record/inversion.pth'

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

        #均设置采用评估模式
        classifier.eval()
        inversion.eval()

    #     with torch.no_grad():  # 禁用梯度计算
    #         for data, target in data_loader:
    #             data, target = data.to(device), target.to(device)
    #             split_data = classifier(data, args.split_point,0,0,args.split_point,args.dropout)
    #             reconstruction = inversion(split_data)
    #
    #             temp_mse_loss = F.mse_loss(reconstruction, data)
    #             layer_record_mse.append(temp_mse_loss.item())
    #         record_mse.append(layer_record_mse)
    #
    #     # 保存mse数组
    # os.makedirs('../mse_plot/', exist_ok=True)
    # mse_dir = '../mse_plot/dropout_split'+str(args.split_point)+'.npy'
    # np.save(mse_dir, record_mse)

    # 预处理snr数组
        with torch.no_grad():  # 禁用梯度计算
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                split_data = classifier(data, args.split_point,0,0,args.split_point,args.dropout)
                reconstruction = inversion(split_data)
                data_numpy = data.cpu().detach().numpy()
                reconstruction_numpy = reconstruction.cpu().detach().numpy()
                snr = SNR(data_numpy, data_numpy - reconstruction_numpy)
                layer_record_mse.append(snr.item())
            record_mse.append(layer_record_mse)

    # #保存snr数组
    os.makedirs('../snr_plot/', exist_ok=True)
    mse_dir = '../snr_plot/dropout_split'+str(args.split_point)+'.npy'
    np.save(mse_dir, record_mse)


def main():
    args = parser.parse_args()

    # 设置是否使用cuda
    cuda = "cuda:" + str(args.cuda_id)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device(cuda if use_cuda else "cpu")
    torch.cuda.set_device(args.cuda_id)
    kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if use_cuda else {}
    # 打印设备信息
    print(f"Selected device: {device}")
    print("Default device ID:", torch.cuda.current_device())  # 应与 args.cuda_id 一致
    print("Default device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # 设置随机数种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # mel_sprect
    mel_spect_acc_100_set = StudioSet(root="../Speech_Command/audio/mel_spect_acc_100", transform=transform)
    mel_spect_acc_all_set = StudioSet(root="../Speech_Command/audio/test_set_mel_spect_numpy", transform=transform)
    mel_spect_acc_100_loader = torch.utils.data.DataLoader(mel_spect_acc_100_set, batch_size=1, shuffle=True, **kwargs)
    mel_spect_acc_all_loader = torch.utils.data.DataLoader(mel_spect_acc_all_set, batch_size=1, shuffle=True, **kwargs)

    args.split_point = 1
    dropout_mse_plot(mel_spect_acc_100_loader, args, device, cuda, '')
    args.split_point = 3
    dropout_mse_plot(mel_spect_acc_100_loader, args, device, cuda, '')

    # os.makedirs('../SNR/SpeechCommand_dropout/', exist_ok=True)
    # os.makedirs('../acc/SpeechCommand_dropout/', exist_ok=True)
    #
    # dropout = [0, 1, 2, 3, 4, 5]
    # split=[1,3]
    # for j in range(0, 2):
    #     for i in range(0, 6):
    #         args.dropout = dropout[i]
    #         args.split_point = split[j]
    #
    #         print("================================")
    #         print(args)
    #         print("================================")
    #
    #         # #计算SNR
    #         # mel_spect_acc_100_target_dir = '../SNR/SpeechCommand_dropout/split' + str(args.split_point) + '_dropout'+str(args.dropout)+'_100.txt'
    #         # mel_spect_acc_all_target_dir = '../SNR/SpeechCommand_dropout/split' + str(args.split_point) + '_dropout'+str(args.dropout)+'_all.txt'
    #         # dropout_SNR(mel_spect_acc_100_loader, args, device, cuda, mel_spect_acc_100_target_dir)
    #         # dropout_SNR(mel_spect_acc_all_loader, args, device, cuda, mel_spect_acc_all_target_dir)
    #
    #         # 计算acc
    #         mel_spect_acc_100_target_dir = '../acc/SpeechCommand_dropout/split' + str(args.split_point) + '_dropout' + str(args.dropout) + '_100.txt'
    #         mel_spect_acc_all_target_dir = '../acc/SpeechCommand_dropout/split' + str(args.split_point) + '_dropout' + str(args.dropout) + '_all.txt'
    #         dropout_acc(mel_spect_acc_100_loader, args, device, cuda, mel_spect_acc_100_target_dir)
    #         dropout_acc(mel_spect_acc_all_loader, args, device, cuda, mel_spect_acc_all_target_dir)




if __name__ == '__main__':
    main()
