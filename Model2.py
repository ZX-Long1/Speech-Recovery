import torch
import torch.nn as nn  #包含构建神经网络的所有构建块，如全连接层、卷积层、循环层、激活函数层
import torch.nn.functional as F  #导入PyTorch的nn.functional模块，包含了定义在神经网络中的大多数非线性激活函数、损失函数等

class Classifier(nn.Module):#继承module基类，定义网络
    #约定类中函数第一个参数为self 表示该类的一个实例对象
    def __init__(self,nc,ndf,nz,select_layer=0, frac=1):
        #调用父类的构造函数——子类调用父类的初始化方法初始化继承的属性
        super(Classifier,self).__init__()
        #nc：输入图像的通道数
        #ndf：某层卷积核的个数 即输出通道数
        #nz：用于确定全连接层的输出大小
        self.nc=nc
        self.ndf=ndf
        self.nz=nz
        self.select_layer = select_layer
        self.frac = frac

        #64*200
        #输入通道数 输出通道数 卷积核大小 步长 填充
        self.conv11=nn.Conv2d(nc,ndf,3,1,1)
        # 输出归一化，基于ndf个特征图进行，有助于加速训练过程并减少过拟合
        self.bn11=nn.BatchNorm2d(ndf)
        #参数 bool值 True时替换原变量的值；False时则保留原输入值
        self.relu11=nn.ReLU(True)
        # print("输出通道大小11",ndf)

        self.conv12=nn.Conv2d(ndf,ndf,3,1,1)
        self.bn12=nn.BatchNorm2d(ndf)
        self.relu12=nn.ReLU(True)
        # print("输出通道大小12", ndf)

        self.conv13=nn.Conv2d(ndf,ndf,3,1,1)
        self.bn13=nn.BatchNorm2d(ndf)
        self.relu13 = nn.ReLU(True)
        self.pool1=nn.MaxPool2d(2,2,0)# ndf*32*100
        # print("输出通道大小13", ndf)

        #32*100
        self.conv21=nn.Conv2d(ndf,ndf*2,3,1,1)
        self.bn21=nn.BatchNorm2d(ndf*2)
        self.relu21=nn.ReLU(True)  #(ndf*2)*32*100
        # print("输出通道大小21", ndf*2)

        self.conv22=nn.Conv2d(ndf*2,ndf*2,3,1,1)
        self.bn22=nn.BatchNorm2d(ndf*2)
        self.relu22 = nn.ReLU(True)
        self.pool2=nn.MaxPool2d(2,2,0) # (ndf*2)*16*50
        # print("输出通道大小22", ndf*2)

        #16*50
        self.conv31=nn.Conv2d(ndf*2,ndf*4,3,1,1)
        self.bn31=nn.BatchNorm2d(ndf*4)
        self.relu31=nn.ReLU(True)  #(ndf*4)*16*50
        # print("输出通道大小31", ndf * 4)

        self.conv32=nn.Conv2d(ndf*4,ndf*4,3,1,1)
        self.bn32=nn.BatchNorm2d(ndf*4)
        self.relu32=nn.ReLU(True)
        self.pool3 = nn.MaxPool2d(2, 2, 0)#(ndf*4)*8*25
        # print("输出通道大小32", ndf * 4)

        #8*25
        #定义全连接层
        self.fc1=nn.Linear(4*ndf*8*25,ndf*8) #第一层全连接层
        #防止过拟合  丢弃，是一种正则化技术。通过随机关闭一部分神经元，防止过于依赖特定的输入
        self.drop1=nn.Dropout(0.3)
        self.fc2=nn.Linear(ndf*8,ndf)  #第二层全连接层
        self.drop2=nn.Dropout(0.2)
        self.fc3=nn.Linear(ndf,nz)

        # self.fc1 = nn.Linear(4 * ndf * 8 * 25, 4*ndf * 8)  # 第一层全连接层
        # # 防止过拟合  丢弃，是一种正则化技术。通过随机关闭一部分神经元，防止过于依赖特定的输入
        # self.drop1 = nn.Dropout(0.4)
        # self.fc2 = nn.Linear(4*ndf * 8, ndf)  # 第二层全连接层
        # self.drop2 = nn.Dropout(0.4)
        # self.fc3 = nn.Linear(ndf, nz)

    '''补充知识：
    一.分类决策：
    Softmax函数：全连接层的输出通常会传递给Softmax函数进行多分类问题的处理。
                Softmax函数能够将全连接层的输出转换为概率分布，即每个类别的预测概率。
                使得模型能够给出属于各个类别的置信度。
    损失函数与优化：为了训练模型，需要定义一个损失函数来评估模型的预测结果与实际标签之间的差异。
                 在分类问题中，常用的损失函数包括交叉熵损失（Cross-Entropy Loss）等。
                 通过反向传播算法，模型根据损失函数的梯度来更新权重和偏置，以最小化损失并优化分类性能。
    二.结果输出
    经过训练和优化后，模型能够对新的输入图像进行分类预测。
    全连接层的输出经过Softmax函数处理后，可以得到每个类别的预测概率，模型最终会选择概率最高的类别作为分类结果输出。
    '''
    #参数说明 x：输入数据  softmax：采用分类模式  conv，relu，fc:分割位置
    def forward(self,x,split_point=0):
        #view函数,用于改变张量形状，不会改变张量的数据(注意：使用view函数时，需要确保新形状的元素总数与原张量的元素总数相同)
        #-1表示该维度大小自动计算，以便保持元素总数不变  表示batch_size
        x=x.view(-1,1,64,200)

        #第一个block的计算
        x=self.conv11(x)
        x=self.bn11(x)
        x=self.relu11(x)
        if split_point==1:
            return x

        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)

        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu13(x)
        x = self.pool1(x)
        if split_point==2:
            return x

        # 第二个block的计算
        x = self.conv21(x)
        x = self.bn21(x)
        x = self.relu21(x)
        x = self.conv22(x)
        x = self.bn22(x)
        x = self.relu22(x)
        if split_point == 3:
            return x
        x = self.pool2(x)

        # 第三个block的计算
        x = self.conv31(x)
        x = self.bn31(x)
        x = self.relu31(x)
        x = self.conv32(x)
        x = self.bn32(x)
        x = self.relu32(x)
        if split_point == 4:
            return x
        x = self.pool3(x)

        #全连接层输出
        x=x.view(-1,4*self.ndf*8*25)
        x=self.fc1(x)
        x=self.drop1(x)
        x = x.view(-1, self.ndf*8)
        x=self.fc2(x)
        x=self.drop2(x)
        x = x.view(-1, self.ndf)
        x=self.fc3(x)

        return F.log_softmax(x,dim=1)

'''
补充知识：
Conv2d：
功能：执行常规的二维卷积操作。
目的：将高维特征映射转换为低维特征映射，通常用于提取图像中的局部特征。
ConvTranspose2d：
功能：执行二维转置卷积操作，也称为反卷积或上采样卷积。
目的：将低维特征映射转换为高维特征映射，通常用于图像生成、图像分割以及超分辨率重建等任务中，以实现特征图的上采样或尺寸增大。
'''
#sp1 relu11 64*64*200
class Conv_sp1(nn.Module):
    def __init__(self):
        super(Conv_sp1, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x

#sp2 relu13 64*32*100
class Conv_sp2(nn.Module):
    def __init__(self):
        super(Conv_sp2, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Tanh(),

            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x


# sp3 relu22 128*32*100
class Conv_sp3(nn.Module):
    def __init__(self):
        super(Conv_sp3, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 2 * 128, 3, 1, 1),
            nn.BatchNorm2d(2 * 128),
            nn.Tanh(),

            nn.Conv2d(2*128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.Tanh(),

            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x


# sp4 relu32 256*16*50
class Conv_sp4(nn.Module):
    def __init__(self):
        super(Conv_sp4, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256*4, 4, 2, 1),
            nn.BatchNorm2d(256*4),
            nn.Tanh(),

            nn.Conv2d(256*4, 256*4, 3, 1, 1),
            nn.BatchNorm2d(256*4),
            nn.Tanh(),

            nn.ConvTranspose2d(256*4, 256 * 2, 4, 2, 1),
            nn.BatchNorm2d(256*2),
            nn.Tanh(),

            nn.Conv2d(256*2, 256*2, 3, 1, 1),
            nn.BatchNorm2d(256*2),
            nn.Tanh(),

            nn.Conv2d(256 * 2, 256, 3, 1, 1),
            nn.BatchNorm2d(256 ),
            nn.Tanh(),

            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),

            nn.Conv2d(128, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = x.view(-1, self.nz, 64, 64)
        x = self.decoder(x)
        return x

# class Classifier_Origin(nn.Module):
#     def __init__(self):
#         super(Classifier_Origin,self).__init__()
#         nc=1
#         ndf=64
#         nz=10
#         self.conv11 = nn.Conv2d(nc, ndf, 3, 1, 1)
#         self.bn11 = nn.BatchNorm2d(ndf)
#         self.relu11 = nn.ReLU()
#
#         self.conv12 = nn.Conv2d(ndf, ndf, 3, 1, 1)
#         self.bn12 = nn.BatchNorm2d(ndf)
#         self.relu12 = nn.ReLU()
#
#         self.conv13 = nn.Conv2d(ndf, ndf, 3, 1, 1)
#         self.bn13 = nn.BatchNorm2d(ndf)
#         self.relu13 = nn.ReLU()
#         self.pool1 = nn.MaxPool2d(2, 2, 0)  # ndf*32*100
#
#         # 32*100
#         self.conv21 = nn.Conv2d(ndf, ndf * 2, 3, 1, 1)
#         self.bn21 = nn.BatchNorm2d(ndf * 2)
#         self.relu21 = nn.ReLU()  # (ndf*2)*32*100
#
#         self.conv22 = nn.Conv2d(ndf * 2, ndf * 2, 3, 1, 1)
#         self.bn22 = nn.BatchNorm2d(ndf * 2)
#         self.relu22 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2, 2, 0)  # (ndf*2)*16*50
#
#         # 16*50
#         self.conv31 = nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1)
#         self.bn31 = nn.BatchNorm2d(ndf * 4)
#         self.relu31 = nn.ReLU()  # (ndf*4)*16*50
#
#         self.conv32 = nn.Conv2d(ndf * 4, ndf * 4, 3, 1, 1)
#         self.bn32 = nn.BatchNorm2d(ndf * 4)
#         self.relu32 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(2, 2, 0)  # (ndf*4)*8*25
#
#         # 8*25
#         # 定义全连接层
#         self.fc1 = nn.Linear(4 * ndf * 8 * 25, 4 * ndf * 8)  # 第一层全连接层
#         self.drop1 = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(4 * ndf * 8, ndf)  # 第二层全连接层
#         self.drop2 = nn.Dropout(0.4)
#         self.fc3 = nn.Linear(ndf, nz)
#
#
#     #参数说明 x：输入数据  softmax：采用分类模式  conv，relu，fc:分割位置
#     def forward(self,x,split_point=0):
#         x = x.view(-1, 1, 64, 200)
#
#         # 第一个block的计算
#         x = self.conv11(x)
#         x = self.bn11(x)
#         x = self.relu11(x)
#         if split_point == 1:
#             return x
#
#         x = self.conv12(x)
#         x = self.bn12(x)
#         x = self.relu12(x)
#
#         x = self.conv13(x)
#         x = self.bn13(x)
#         x = self.relu13(x)
#         x = self.pool1(x)
#         if split_point == 2:
#             return x
#
#         # 第二个block的计算
#         x = self.conv21(x)
#         x = self.bn21(x)
#         x = self.relu21(x)
#         x = self.conv22(x)
#         x = self.bn22(x)
#         x = self.relu22(x)
#         if split_point == 3:
#             return x
#         x = self.pool2(x)
#
#         # 第三个block的计算
#         x = self.conv31(x)
#         x = self.bn31(x)
#         x = self.relu31(x)
#         x = self.conv32(x)
#         x = self.bn32(x)
#         x = self.relu32(x)
#         if split_point == 4:
#             return x
#         x = self.pool3(x)
#
#         # 全连接层输出
#         x = x.view(-1, 4 * self.ndf * 8 * 25)
#         x = self.fc1(x)
#         x = self.drop1(x)
#         x = x.view(-1, 4 * self.ndf * 8)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         x = x.view(-1, self.ndf)
#         x = self.fc3(x)
#
#         return F.log_softmax(x, dim=1)


# def main():
#     myclassfier=Classifier(1,64,10,0,0)
#     rand_tensor1 = torch.rand(1, 64, 64, 200)   #批量大小，通道数，高度，宽度
#     rand_tensor2 = torch.randn(1, 64, 32, 100)
#     rand_tensor3 = torch.randn(1, 128,32, 100)
#     rand_tensor4 = torch.randn(1, 256,16, 50)
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
#     my_conv_sp3 = Conv_sp3()
#     x3 = my_conv_sp3.decoder(rand_tensor3)
#     print(x3.shape)
#
#     # new_rand_tensor = myclassfier.forward(rand_tensor,0)
#     # print("ans",new_rand_tensor)
#     # print("代码成功运行")
#
# # 按装订区域中的绿色按钮以运行脚本。
# if __name__ == '__main__':
#     main()



