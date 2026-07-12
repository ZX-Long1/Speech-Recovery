from __future__ import print_function
# import argparse                      #用于解析命令行参数
# import torch                         #用于构建和训练神经网络
# import torch.optim as optim
# import torch.nn as nn
# from torchvision import transforms
# import torch.nn.functional as F      #用于图像预处理
# from speech_command_dataset2 import StudioSet as StudioSet #数据集
# from speech_command_Model2 import BasicBlock,ResNet18,Conv_sp4, Conv_sp3, Conv_sp2, Conv_sp1  # 分类器，反转模型
import numpy as np
# import random
# import os
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns


def SNR(signal,noise):
    power_signal = np.sum(signal**2)/signal.size
    power_noise=np.sum(noise**2)/noise.size
    snr=10*np.log10(power_signal/power_noise)
    # print(signal.size,noise.size)
    return snr
#
# def UrbanSound8K_melspect_whitebox_SNR():
#     print()
#
# def UrbanSound8K_melspect_blackbox_SNR():
#     print()
#
# def UrbanSound8K_mfcc_whitebox_SNR():
#     print()
#
# def UrbanSound8K_mfcc_blackbox_SNR():
#     print()
#
# def SpeechCommand_melspect_whitebox_SNR():
#     print()
#
# def SpeechCommand_melspect_blackbox_SNR():
#     print()
#
# def SpeechCommand_mfcc_whitebox_SNR():
#     print()
#
# def SpeechCommand_mfcc_blackbox_SNR():
#     print()
#
# def main():
#     # signal=np.random.rand(3,4)
#     # print(signal)
#     # noise=np.random.rand(3,4)
#     # print(noise)
#     # print(signal-noise)
#     # snr=SNR(signal,signal-noise)
#     # print(snr)

# if __name__ == '__main__':
#     main()


