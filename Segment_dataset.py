from __future__ import print_function,division
import os           #os模块，提供与操作系统交互的功能，比如文件路径操作、环境变量访问、进程管理等。
import numpy as np  #导入numpy库，并将其简称为np。
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from librosa.core.audio import samplerate
from torch.utils.data import Dataset  #从torch.utils.data模块中导入Dataset类。Dataset是PyTorch中用于表示数据集的抽象类，用户需要继承这个类并实现__getitem__和__len__方法，以便自定义数据集
from PIL import Image  #从PIL中导入Image模块。PIL是一个强大的图像处理库，Image模块提供了打开、操作和保存多种格式图像的功能,提供了许多用于图像增强的函数，如裁剪、旋转、缩放、归一化等。
from torchvision import transforms
import scipy.interpolate as sci
import librosa          #用于音频处理和特征提取
import librosa.display  # 用于音频特征的可视化
import matplotlib.pyplot as plt  #导入matplotlib.pyplot模块，并将其简称为plt。绘图库，pyplot提供了一个类似于MATLAB的绘图系统。
import re  #导入re模块，提供了正则表达式的相关功能
import glob  #导入glob模块，它提供了从目录通配符搜索中生成文件列表的函数
import sys
import IPython.display as ipd
from tqdm import tqdm
from cv2 import resize


class StudioSet(Dataset):  # 继承Dateset 加载和预处理语音数据集
    def __init__(self, root, transform=None):
        # 构造函数
        # root:声音数据地址
        # transform:指定变换操作

        studio = []  # 存储声音数据
        target = []  # 存储标签数据
        self.transform = transform

        # 统一Mel频谱矩阵行数
        n_mels = 64
        sample_rate = 49050
        maxcolumn = 96

        # #计算全局最小值和最大值
        # max_val=float('-inf')
        # min_val=float('inf')
        # for wavename in os.listdir(root):
        #     wave_path=os.path.join(root,wavename)
        #     mel_spect=np.load(wave_path,allow_pickle=True)       #加载numpy矩阵
        #     temp_min_val = np.min(mel_spect)
        #     temp_max_val = np.max(mel_spect)
        #     max_val=max(max_val,temp_max_val)
        #     min_val=min(min_val,temp_min_val)
        # # print(max_val) #3.814697265625e-06
        # # print(min_val) #-80.00000381469727

        # 全局标准化
        max_val = 3.814697265625e-06
        min_val = -80.00000381469727

        # for wavename in os.listdir(root):
        #     wave_path = os.path.join(root, wavename)
        #     try:
        #         mel_spect = np.load(wave_path, allow_pickle=True)  # 加载numpy矩阵
        #     except:
        #         continue  # 直接看下一个文件

        for wavename in os.listdir(root):
            wave_path = os.path.join(root, wavename)
            mel_spect = np.load(wave_path, allow_pickle=True)  # 加载numpy矩阵

            # 应用归一化公式
            normalized_mel_spect = ((mel_spect - min_val) / (max_val - min_val)) * 255

            # 迁移学习修改尺寸
            maxcolumn=200
            normalized_mel_spect = resize(normalized_mel_spect, (200, 64))
            # print( normalized_mel_spect.shape)

            # 确保数据类型为uint8
            mel_spect = np.uint8(normalized_mel_spect)

            studio.append(mel_spect)  # 存储mel_spect图像数据
            label = int(wavename.split('_')[0])  # 音频对应的类别标签
            target.append(label)  # 存储标签数据
            # print(label,wavename)

            # # 绘制 Mel 谱图
            # plt.figure(figsize=(6, 4))
            # librosa.display.specshow(mel_spect, sr=sample_rate, x_axis='time', y_axis='mel')
            # plt.ylabel('Mel Frequency')
            # plt.xlabel('Time (s)')
            # plt.title(f'Mel Spectrogram of {wavename}')
            # plt.colorbar(format='%+2.0f dB')  # 添加色条，显示dB值
            # plt.show()

        # print(len(studio))
        # print(len(target))
        # #沿指定轴连接数组序列
        # self.data=np.concatenate(studio,axis=0)

        self.data = np.array(studio)
        # 将列表转换为NumPy数组
        self.label = np.array(target)

        # 将数据及其对应标签随机打乱
        np.random.seed(66)
        perm = np.arange(len(self.data))  # 生成一个0到len(self.data-1)的整数数组
        np.random.shuffle(perm)  # 随机打乱作为新的索引
        self.data = self.data[perm]  # 根据新的索引顺序重排
        self.label = self.label[perm]

    def __getitem__(self, index): #根据索引 获取样本
        studio=self.data[index]
        label=self.label[index]

        if self.transform is not None and callable(self.transform):
            # 检查是否需要从numpy数组转换为PIL图像
            if not isinstance(studio, Image.Image):
                studio = Image.fromarray(studio)  # 将numpy数组转换为PIL图像
            studio = self.transform(studio)  # 应用转换

        return (studio,label) #返回对应的mfcc图像和标签

    def __len__(self):  # 返回数据集中的样本总数
        return len(self.data)


# source_folders="../Speech_Command/audio"
# train_set_path = os.path.join(source_folders, 'train_set_mel_spect_numpy')  # 训练数据地址
# test_set_path = os.path.join(source_folders, 'test_set_mel_spect_numpy')  # 测试数据地址
# # print(train_set_path)
#
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
#
# train_set = StudioSet(train_set_path,transform)
# print("train_set length:", len(train_set))
# test_set=StudioSet(test_set_path,transform)
# print("test_set length:", len(test_set))
#
# # test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
# # for batch_idx, (data, target) in enumerate(test_loader):
# #     if batch_idx > 5:
# #         break
# #     print(batch_idx)
# #     print(data)
# #     print(target)



