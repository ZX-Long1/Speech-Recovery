import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


'''
总变差（Total Variation, TV）的计算、基于高斯权重的SSIM（Structural Similarity Index Measure）的计算，以及相关的辅助函数。
总变差（TV）
  TV 函数：函数计算输入图像 x 的总变差，即图像在水平和垂直方向上的梯度之和。首先将图像在水平和垂直方向上的相邻像素之间的差异平方并求和，然后除以相应的像素数（忽略边缘像素）和批次大小来归一化。
  TV_beta1 函数：这个函数的实现似乎有误。它试图计算某种形式的总变差，但方式并不正确。

高斯权重：gaussian 函数生成一维高斯权重，而 create_window 函数则利用这些权重创建一个二维的高斯核，用于SSIM计算中的局部平均和标准差估计。这是SSIM算法中的一个重要步骤，用于模拟视觉系统对图像局部结构的感知。
  SSIM 辅助函数：_ssim 函数实现了SSIM的核心计算。它使用高斯核来估计两个图像的局部均值 mu1 和 mu2，以及它们的平方 mu1_sq 和 mu2_sq、乘积 mu1_mu2、方差 sigma1_sq 和 sigma2_sq 以及协方差 sigma12。然后，它使用这些统计量来计算SSIM指数。
  SSIM 类：SSIM 类封装了SSIM计算，使其可以像PyTorch模块一样使用。它接受一个窗口大小（默认为11）和一个大小平均（默认为True）的参数。
          在 forward 方法中，它会检查输入图像的通道数是否与当前窗口的通道数匹配，并根据需要创建新的窗口。这种设计允许SSIM模块处理不同通道数的图像，
          尽管在这个实现中，channel 初始化为1，可能意味着它默认用于灰度图像。
'''

def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])

    h_tv = torch.pow(x[:, :, 1:, :]-x[:, :, :h_x-1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:]-x[:, :, :, :w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


# 这个感觉有一点问题，就是矩阵幂次乘法那里
def TV_beta1(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    pic1 = torch.pow(x[:, :, :, :],2)
    pic2 = torch.pow(x[:, :, :, :],2)
    h_tv = torch.pow(x[:, :, 1:, :]-x[:, :, :h_x-1, :], 2)
    w_tv = torch.pow(x[:, :, :, 1:]-x[:, :, :, :w_x-1], 2)
    pic1[:, :, 1:, :] = h_tv
    pic2[:, :, :, 1:] = w_tv
    tmp = torch.pow(pic1 + pic2, 0.5).sum()
    return tmp / (batch_size*count_h)

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def speech_tf_loss(M, alpha=1.0):
    temporal_loss = torch.mean(torch.abs(
        M[:, :, :, 2:] - 2 * M[:, :, :, 1:-1] + M[:, :, :, :-2]))
    spectral_loss = torch.mean(torch.abs(
        M[:, :, 1:, :] - M[:, :, :-1, :]))
    speech_reg = temporal_loss + alpha * spectral_loss
    return speech_reg, temporal_loss, spectral_loss