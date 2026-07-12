import torch
import torch.nn as nn  #包含构建神经网络的所有构建块，如全连接层、卷积层、循环层、激活函数层
import torch.nn.functional as F  #导入PyTorch的nn.functional模块，包含了定义在神经网络中的大多数非线性激活函数、损失函数等

#残差块 每个残差块包含两个卷积层和一个捷径连接（shortcut)
class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet18(nn.Module):
    def __init__(self, BasicBlock, nc,nz):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,64,[[1,1],[1,1]])
        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])
        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])
        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,512,[[2,1],[1,1]])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, nz)

    #这个函数主要是用来，重复同一个残差块
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x,split_point=0):
        out = self.conv1(x)
        if split_point==1:
            return out
        out = self.conv2(out)
        if split_point==2:
            return out
        out = self.conv3(out)
        if split_point==3:
            return out
        out = self.conv4(out)
        if split_point==4:
            return out
        out = self.conv5(out)

        # out = F.avg_pool2d(out,7)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return F.log_softmax(out,dim=1)

# sp1 64*16*24
class Conv_sp1(nn.Module):
    def __init__(self):
        super(Conv_sp1, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x24 -> 32x48
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 32x48 -> 64x96
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

#sp2 64*16*24
class Conv_sp2(nn.Module):
    def __init__(self):
        super(Conv_sp2, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1),  # 16x24 -> 32x48
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),  #  32x48
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x48 -> 64x96
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # 64x96
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=1, padding=1),   # 64x96
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

# sp3 128*8*12
class Conv_sp3(nn.Module):
    def __init__(self):
        super(Conv_sp3, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=4, stride=2, padding=1),  # 8x12 -> 16x24
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # 16x24 -> 32x48
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),  # 32x48
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1), # 32x48 -> 64x96
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),  # 64x96
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # 64x96
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=1, padding=1),  # 64x96
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

# sp4  256*4*6
class Conv_sp4(nn.Module):
    def __init__(self):
        super(Conv_sp4, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 512, kernel_size=4, stride=2, padding=1), # 4x6 -> 8x12
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 8x12 -> 16x24
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, 1024, kernel_size=4, stride=2, padding=1),  # 16*24 -> 32x48
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1),  # 32x48 -> 64*96
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1024, 256, kernel_size=3, stride=1, padding=1),  # 64x96
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),  # 64x96
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 1, kernel_size=3, stride=1, padding=1),  # 64x96
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x
#
# def main():
#     res18 = ResNet18(BasicBlock,1,35)
#     out=res18(torch.randn(1,1,64,96),split_point=1) #64 16*24
#     print(out.shape)
#     # print(out)
#     out=res18(torch.randn(1,1,64,96),split_point=2) #64 16*24
#     print(out.shape)
#     # print(out)
#     out=res18(torch.randn(1,1,64,96),split_point=3) #128 8*12
#     print(out.shape)
#     # print(out)
#     out=res18(torch.randn(1,1,64,96),split_point=4) #256 4*6
#     print(out.shape)
#     # print(out)
#     out = res18(torch.randn(1, 1, 64, 96))
#     print(out.shape)
#     # print(out)
#
#
#     rand_tensor1 = torch.rand(1, 64, 16, 24)   #批量大小，通道数，高度，宽度
#     rand_tensor2 = torch.randn(1, 64, 16, 24)
#     rand_tensor3 = torch.randn(1, 128,8, 12)
#     rand_tensor4 = torch.randn(1, 256,4, 6)
#
#     my_conv_sp1=Conv_sp1()
#     x1=my_conv_sp1.decoder(rand_tensor1)
#     print(x1.shape)
#
#     my_conv_sp2 = Conv_sp2()
#     x2 = my_conv_sp2.decoder(rand_tensor2)
#     print(x2.shape)
#
#     my_conv_sp3 = Conv_sp3()
#     x3 = my_conv_sp3.decoder(rand_tensor3)
#     print(x3.shape)
#
#     my_conv_sp4 = Conv_sp4()
#     x4 = my_conv_sp4.decoder(rand_tensor4)
#     print(x4.shape)
#
#     print("代码成功运行")
#
# # 按装订区域中的绿色按钮以运行脚本。
# if __name__ == '__main__':
#     main()

