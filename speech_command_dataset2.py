from __future__ import print_function,division
import os           #os模块，提供与操作系统交互的功能，比如文件路径操作、环境变量访问、进程管理等。
import numpy as np  #导入numpy库，并将其简称为np。
from torch.utils.data import Dataset  #从torch.utils.data模块中导入Dataset类。Dataset是PyTorch中用于表示数据集的抽象类，用户需要继承这个类并实现__getitem__和__len__方法，以便自定义数据集
from PIL import Image  #从PIL中导入Image模块。PIL是一个强大的图像处理库，Image模块提供了打开、操作和保存多种格式图像的功能,提供了许多用于图像增强的函数，如裁剪、旋转、缩放、归一化等。
from torchvision import transforms
import scipy.interpolate as sci
import librosa          #用于音频处理和特征提取
import librosa.display  # 用于音频特征的可视化
import torch
import matplotlib.pyplot as plt  #导入matplotlib.pyplot模块，并将其简称为plt。绘图库，pyplot提供了一个类似于MATLAB的绘图系统。
import re  #导入re模块，提供了正则表达式的相关功能
import glob  #导入glob模块，它提供了从目录通配符搜索中生成文件列表的函数

#填充时进行一维插值
def pad_or_trim_mfcc(mfcc, max_length=100):
    if mfcc.shape[1] > max_length:
        return mfcc[:, :max_length]
    else:
        pad_width = max_length - mfcc.shape[1]
        padded_mfcc = np.zeros((mfcc.shape[0], max_length))
        padded_mfcc[:, :mfcc.shape[1]] = mfcc

        # 对每一行进行线性插值
        for i in range(mfcc.shape[0]):
            x_new = np.linspace(0, mfcc.shape[1] - 1, max_length)
            x_old = np.arange(mfcc.shape[1])
            f_interp = sci.interp1d(x_old, mfcc[i, :], kind='linear', fill_value='extrapolate')  #对矩阵每一行进行线性填充
            padded_mfcc[i, :] = f_interp(x_new)

        # print(padded_mfcc.shape)
        return padded_mfcc


#获取mfcc矩阵的最大列数
def get_maxcolumn(root,n_mfcc,sample_rate):
    maxcolumn=0
    for wavename in os.listdir(root):
        wave_path = os.path.join(root, wavename)
        y, sr = librosa.load(wave_path, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        if mfccs.shape[1]>maxcolumn:
            maxcolumn=mfccs.shape[1]
    return maxcolumn

def init1():
    # maxcolumn = get_maxcolumn(train_folders, n_mfcc, sample_rate)  # 记录mfcc矩阵的最大维度
    # print('maxcolumn', maxcolumn)
    # maxcolumn = get_maxcolumn(test_folders, n_mfcc, sample_rate)  # 记录mfcc矩阵的最大维度
    # print('maxcolumn', maxcolumn)

    train_folders = "../Speech_Command/audio/train_set"
    test_folders = "../Speech_Command/audio/test_set"
    train_folders_save = "../Speech_Command/audio/train_set_mfcc_numpy"
    test_folders_save = "../Speech_Command/audio/test_set_mfcc_numpy"

    sample_rate = 49050 # 设置采样率
    n_mfcc = 64  # MFCC特征数
    maxcolumn = 96

    #检测对应文件路径是否存在，不存在则创建
    os.makedirs( train_folders_save, exist_ok=True)
    os.makedirs(test_folders_save, exist_ok=True)

    #分别处理训练集和测试集
    for wavename in os.listdir(train_folders):
        wave_path = os.path.join(train_folders, wavename)
        y, sr = librosa.load(wave_path, sr=sample_rate)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = pad_or_trim_mfcc(mfccs, maxcolumn)

        mfcc_file_name = os.path.splitext(wavename)[0] + '.npy'
        mfcc_file_path = os.path.join(train_folders_save, mfcc_file_name)

        np.save(mfcc_file_path, mfccs)
        print(f'Saved MFCC matrix for {wavename} to {mfcc_file_path}') #f前缀在字符串前表示格式化字符串,大括号{}内的内容会被当作表达式进行求值，并且求值结果会转换成字符串插入到该位置。

    for wavename in os.listdir(test_folders):
        wave_path = os.path.join(test_folders, wavename)
        y, sr = librosa.load(wave_path, sr=sample_rate)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = pad_or_trim_mfcc(mfccs, maxcolumn)

        mfcc_file_name = os.path.splitext(wavename)[0] + '.npy'
        mfcc_file_path = os.path.join(test_folders_save, mfcc_file_name)

        np.save(mfcc_file_path, mfccs)
        print(f'Saved MFCC matrix for {wavename} to {mfcc_file_path}')

class StudioSet(Dataset):  # 继承Dateset 加载和预处理语音数据集
    def __init__(self, root, transform=None):
        # 构造函数
        # root:声音数据地址
        # transform:指定变换操作

        studio = []  # 存储声音数据
        target = []  # 存储标签数据
        self.transform = transform

        self.sample_rate = 49050
        self.n_mfcc = 64 #计算MFCC特征，通常取13个系数
        self.maxcolumn = 96
        self.sr=self.sample_rate

        # #计算全局最小值和最大值
        # max_val=float('-inf')
        # min_val=float('inf')
        # for wavename in os.listdir(root):
        #     wave_path=os.path.join(root,wavename)
        #     mfccs=np.load(wave_path,allow_pickle=True)       #加载numpy矩阵
        #     temp_min_val = np.min(mfccs)
        #     temp_max_val = np.max(mfccs)
        #     max_val=max(max_val,temp_max_val)
        #     min_val=min(min_val,temp_min_val)
        # print(max_val) #348.4483642578125
        # print(min_val) #-895.2033081054688

        # 全局标准化
        max_val=348.4483642578125
        min_val=-895.2033081054688

        # os.listdir()以路径作为参数，返回一个包含路径下所有文件和子目录名称的列表
        for wavename in os.listdir(root):
            wave_path=os.path.join(root,wavename)
            mfccs=np.load(wave_path,allow_pickle=True)       #加载numpy矩阵
            # print(mfccs.shape)

            # 应用归一化公式
            normalized_mfccs = ((mfccs - min_val) / (max_val - min_val)) * 255
            # 确保数据类型为uint8
            mfccs = np.uint8(normalized_mfccs)

            studio.append(mfccs) #存储mfcc图像数据
            label=int(wavename.split('_')[0]) #音频对应的类别标签
            target.append(label) #存储标签数据

            # # print(mfccs.shape)
            # # 可视化MFCC特征
            # plt.figure(figsize=(8, 4))
            # librosa.display.specshow(mfccs, sr=self.sr, x_axis='time', y_axis='mel', hop_length=512, cmap='magma')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title('MFCC')
            # plt.xlabel('Time (s)')
            # plt.ylabel('MFCC Frequency')
            # plt.tight_layout()
            # plt.show()

        # print(len(studio))
        # print(len(target))
        # #沿指定轴连接数组序列
        # self.data=np.concatenate(studio,axis=0)

        self.data = np.array(studio)
        #将列表转换为NumPy数组
        self.label=np.array(target)

        #将数据及其对应标签随机打乱
        np.random.seed(66)
        perm=np.arange(len(self.data))  #生成一个0到len(self.data-1)的整数数组
        np.random.shuffle(perm) #随机打乱作为新的索引
        self.data=self.data[perm] #根据新的索引顺序重排
        self.label=self.label[perm]

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


# # init1()
source_folders="../Speech_Command/audio"
train_set_path = os.path.join(source_folders, 'train_set_mfcc_numpy')  # 训练数据地址
test_set_path = os.path.join(source_folders, 'test_set_mfcc_numpy')  # 测试数据地址
# print(train_set_path)
# print(test_set_path)

transform = transforms.Compose([
    transforms.ToTensor(),
])

train_set = StudioSet(train_set_path,transform)
test_set=StudioSet(test_set_path,transform)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
for batch_idx, (data, target) in enumerate(test_loader):
    if batch_idx > 5:
        break
    print(batch_idx)
    print(data)
    print(target)

