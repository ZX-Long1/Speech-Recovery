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


#MFCC裁剪与填充
#填充时直接补零
# def pad_or_trim_mfcc(mfcc, max_length=100):
#     if mfcc.shape[1] > max_length:
#         return mfcc[:, :max_length]
#     else:
#         pad_width = max_length - mfcc.shape[1]
#         return np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

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
    # self.maxcolumn=get_maxcolumn(root,self.n_mfcc,self.sample_rate)#记录mfcc矩阵的最大维度
    # print('maxcolumn',self.maxcolumn)

    sample_rate = 25250  # 设置采样率
    n_mfcc = 64  # 计算MFCC特征，通常取13个系数
    maxcolumn = 200

    train_folders = "..\\UrbanSound8K\\UrbanSound8K\\audio\\train_set"
    test_folders="..\\UrbanSound8K\\UrbanSound8K\\audio\\test_set"
    train_folders_save1 = "..\\UrbanSound8K\\UrbanSound8K\\audio\\train_set_mfccsnumpy"
    test_folders_save1 = "..\\UrbanSound8K\\UrbanSound8K\\audio\\test_set_mfccsnumpy"

    #检测对应文件路径是否存在，不存在则创建
    os.makedirs( train_folders_save1, exist_ok=True)
    os.makedirs(test_folders_save1, exist_ok=True)

    #分别处理训练集和测试集
    for wavename in os.listdir(train_folders):
        wave_path = os.path.join(train_folders, wavename)
        # 加载音频文件
        # librosa.load函数返回两个值：y（音频时间序列）和sr（音频的采样率）
        # 参数sr=None表示加载音频时使用其原始采样率，不进行重采样
        y, sr = librosa.load(wave_path, sr=sample_rate)
        # print(y,sr)

        # librosa.feature.mfcc函数返回MFCC特征矩阵，每一行代表一个不同的MFCC系数，每一列对应音频信号的一个时间帧
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = pad_or_trim_mfcc(mfccs, maxcolumn)
        # print(mfccs.shape)#输出mfccs的维度

        mfcc_file_name = os.path.splitext(wavename)[0] + '.npy'
        mfcc_file_path = os.path.join(train_folders_save1, mfcc_file_name)

        # 保存 MFCC 矩阵
        np.save(mfcc_file_path, mfccs)
        print(f'Saved MFCC matrix for {wavename} to {mfcc_file_path}') #f前缀在字符串前表示格式化字符串,大括号{}内的内容会被当作表达式进行求值，并且求值结果会转换成字符串插入到该位置。

    for wavename in os.listdir(test_folders):
        wave_path = os.path.join(test_folders, wavename)
        y, sr = librosa.load(wave_path, sr=sample_rate)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfccs = pad_or_trim_mfcc(mfccs, maxcolumn)

        mfcc_file_name = os.path.splitext(wavename)[0] + '.npy'
        mfcc_file_path = os.path.join(test_folders_save1, mfcc_file_name)

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

        self.sample_rate = 25250  # 设置采样率
        self.n_mfcc = 64 #计算MFCC特征，通常取13个系数
        self.maxcolumn = 200
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
        # print(max_val) #302.4324184398555
        # print(min_val) #-906.176513671875

        # # 局部标准化
        # # os.listdir()以路径作为参数，返回一个包含路径下所有文件和子目录名称的列表
        # for wavename in os.listdir(root):
        #     wave_path = os.path.join(root, wavename)
        #     mfccs = np.load(wave_path, allow_pickle=True)  # 加载numpy矩阵
        #     # 计算MFCC矩阵的最小值和最大值(不是全局的，只是相对于每个文件来说)
        #     min_val = np.min(mfccs)
        #     max_val = np.max(mfccs)
        #     # 应用归一化公式
        #     normalized_mfccs = ((mfccs - min_val) / (max_val - min_val)) * 255
        #     # 确保数据类型为uint8
        #     mfccs = np.uint8(normalized_mfccs)
        #     studio.append(mfccs)  # 存储mfcc图像数据
        #     label = int(wavename.split('-')[1])  # 音频对应的类别标签
        #     target.append(label)  # 存储标签数据

        # 全局标准化
        max_val=302.4324184398555
        min_val=-906.176513671875

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
            label=int(wavename.split('-')[1]) #音频对应的类别标签
            target.append(label) #存储标签数据

            # print(mfccs.shape)
            # # 可视化MFCC特征
            # plt.figure(figsize=(10, 4))  # 创建图形窗口并指定大小
            # librosa.display.specshow(mfccs, sr=self.sr, x_axis='time')
            # plt.colorbar(format='%+2.0f dB')
            # plt.title('MFCC')
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
# source_folders="..\\UrbanSound8K\\UrbanSound8K\\audio"
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
# # train_set_path = os.path.join(source_folders, 'train_set')  # 训练数据地址
# train_set_path = os.path.join(source_folders, 'train_set_mfccsnumpy')  # 训练数据地址
# transform = transforms.Compose([
#     transforms.ToTensor(),
# ])
# train_set = StudioSet(train_set_path,transform)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
# for batch_idx, (data, target) in enumerate(train_loader):
#     if batch_idx > 5:
#         break
#     print(batch_idx)
#     # print(data)
#     # print(target)
#     mfccs = data[0].cpu()
#     mfccs = mfccs.squeeze(0)
#     print(mfccs.shape)
#     mfccs_numpy = mfccs.numpy()
#     print(mfccs_numpy.shape)
#
#     # 可视化MFCC特征
#     sr = 25250  # 设置采样率
#     plt.figure(figsize=(8, 4))
#     librosa.display.specshow(mfccs_numpy, sr=sr, x_axis='time', y_axis='mel',hop_length=512,cmap = 'magma')
#     plt.title('MFCC')
#     plt.colorbar(format='%+2.0f dB')
#     plt.tight_layout()
#     plt.show()



'''
从wav类型语音进行MFCC特征提取，并使用CNN训练语音分类模型的全过程，可以细分为以下几个关键步骤：

一、预处理阶段
读取音频文件：
使用Python中的librosa库读取wav格式的音频文件。librosa是一个用于音频和音乐分析的Python库，支持多种音频和音乐分析功能。
python
import librosa  
y, sr = librosa.load('audio.wav', sr=None)  # 读取音频文件，sr=None表示使用音频文件的原始采样率
预加重（可选）：
对音频信号进行预加重处理，以减少信号中的高频噪声。这一步虽然不是MFCC特征提取的必需步骤，但在某些情况下可以改善特征提取的效果。
分帧：
将音频信号分割成多个固定长度的帧。每帧通常包含几十到几百毫秒的音频数据，并且帧之间可以有一定的重叠。
加窗：
对每一帧进行加窗处理，常用的窗函数有汉明窗、汉宁窗等。加窗的目的是为了减少帧两端的信号不连续性。

二、MFCC特征提取
快速傅里叶变换（FFT）：
对加窗后的每一帧信号进行FFT，得到频谱图。FFT是信号处理中常用的算法，可以将信号从时域转换到频域。
梅尔滤波器组：
将频谱图通过一组梅尔滤波器进行滤波，得到梅尔频谱图。梅尔滤波器组是根据人耳的听觉特性设计的，能够更好地模拟人耳对频率的感知。
对数运算：
对梅尔频谱图的每个频带取对数运算，得到对数梅尔频谱图。这一步是为了进一步压缩频谱的动态范围，使特征更加平滑。
离散余弦变换（DCT）：
对对数梅尔频谱图进行DCT变换，得到MFCC系数。DCT变换可以将信号从时域或频域转换到另一种正交域，同时保留信号的主要特征。
python
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # 提取MFCC特征，n_mfcc为提取的MFCC系数数量


三、准备训练数据
数据标准化/归一化：
对提取的MFCC特征进行标准化或归一化处理，以提高模型的训练效率和性能。
构造特征矩阵：
将多个音频文件的MFCC特征按批次（batch）组织成特征矩阵，同时准备对应的标签（label）矩阵。


四、构建CNN模型
设计CNN架构：
根据任务需求设计CNN的架构，包括卷积层（Convolutional Layer）、池化层（Pooling Layer）、全连接层（Fully Connected Layer）等。
在语音识别任务中，由于输入数据是二维的（帧×系数），因此可以使用二维卷积层或一维卷积层（如果考虑时间序列特性）。
设置模型参数：
确定卷积核的大小、数量、步长等参数，以及池化层的类型、大小等。
设定损失函数（如交叉熵损失函数）和优化器（如Adam优化器）。

五、训练模型
训练模型：
使用准备好的训练数据对CNN模型进行训练。训练过程中，模型会通过反向传播算法自动调整权重和偏置，以最小化损失函数。
评估模型：
使用测试数据集评估模型的性能，包括准确率、召回率、F1分数等指标。

六、模型优化与部署
模型优化：
根据评估结果对模型进行调优，包括调整模型架构、参数、正则化方法等。
可以使用模型融合、集成学习等技术进一步提高模型的性能。
模型部署：
将训练好的模型部署到实际应用中，如语音识别系统、语音助手等。
以上就是从wav类型语音进行MFCC特征提取，并使用CNN训练语音分类模型的全过程。需要注意的是，实际操作中可能需要根据具体的数据集和任务需求进行适当的调整和优化。
'''