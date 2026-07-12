#Speech_Command_Classifier 针对音频数据形式mel_spect

from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from torchvision import transforms
import os
import torch.nn.functional as F
from speech_command_dataset1 import StudioSet
# from speech_command_Model import BasicBlock,ResNet18
from speech_command_Model2 import BasicBlock,ResNet18
import numpy as np
import random

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='') #指定每个批次中的样本数量
parser.add_argument('--test-batch-size', type=int, default=128, metavar='') #指定测试集上每个批次中的样本数量
parser.add_argument('--epochs', type=int, default=125, metavar='') #指定整个训练数据集被遍历的次数
parser.add_argument('--lr', type=float, default=0.0002, metavar='') #学习率，控制参数更新的步长大小 越小的学习率参数调整越细致，但会导致训练过程缓慢
parser.add_argument('--momentum', type=float, default=0.5, metavar='')  #动量参数，用于加速SGD在相关方向上的优化，并抑制震荡
parser.add_argument('--no-cuda', action='store_true', default=False)  #指定是否进行CUDA训练
parser.add_argument('--seed', type=int, default=1, metavar='') #随机数生成器的种子，保证实验的可重复性
parser.add_argument('--log-interval', type=int, default=50, metavar='') #指定打印间隔，即指定多少批次后打印一次日志信息
parser.add_argument('--nc', type=int, default=1) #表示输入图像的通道数
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nz', type=int, default=35)
parser.add_argument('--num_workers', type=int, default=50, metavar='') #指定数据加载的工作进程数  增加工作进程数可以加快数据加载过程，但可能会增加内存消耗
parser.add_argument('--cuda_id', type=int, default=2, metavar='')
parser.add_argument('--split_point', type=int, default=0, metavar='')

def train(classifier, log_interval, device, data_loader, optimizer, epoch,target_dir):
    classifier.train() #设置模型为训练模式
    #遍历数据加载器 按批次获取数据和目标标签
    #batch_idx是批次的索引  (data,target)是对应批次的数据和标签  enumerate是遍历数据加载器
    for batch_idx, (data, target) in enumerate(data_loader):
        if target.dtype != torch.long:
            target = target.long()  # 转换为长整型

        data, target = data.to(device), target.to(device)  #将数据移动到device
        optimizer.zero_grad()  #在每次迭代开始前，清零梯度。PyTorch会累积梯度，因此需要初始化
        output = classifier(data)  #前向传播计算

        #注意，当使用负对数似然损失作为损失函数时，output应该是对数概率，即经过log_softmax
        loss = F.nll_loss(output, target)  #计算负对数损失
        loss.backward()  #反向传播，计算损失函数关于模型参数的梯度
        optimizer.step()  #更新模型的参数

        #日志记录如果当前批次索引是log_interval的倍数，则打印当前的训练轮次、已处理的样本数、总样本数和当前损失。
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data),
                                                                  len(data_loader.dataset), loss.item()))

def test(classifier, device, data_loader,target_dir):
    classifier.eval()  #设置模型为评估模式
    test_loss = 0  #初始化测试损失和正确预测计数
    correct = 0
    with torch.no_grad():  #禁用梯度计算，即模型评估时不需要进行反向传播
        for data, target in data_loader:
            if target.dtype != torch.long:
                target = target.long()  # 转换为长整型

            data, target = data.to(device), target.to(device)
            output = classifier(data)

            #使用负对数似然损失计算当前损失  并将损失累加
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            #根据预测概率获取预测类别
            pred = output.max(1, keepdim=True)[1]
            #将预测结果与真实标签比较
            #sum方法计算正确预测的样本数，item方法进行转换为python标量
            correct += pred.eq(target.view_as(pred)).sum().item()

    #计算平均测试损失和准确率
    test_loss /= len(data_loader.dataset)

    with open(target_dir + 'record_test.txt', 'a') as file:
        file.write('\nTest classifier: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    print('\nTest classifier: Average loss: {:.6f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
    return correct / len(data_loader.dataset)

def main():
    #命令行参数解析
    args = parser.parse_args()
    print("================================")
    print(args)
    print("================================")

    target_dir="../SplitLearn/Speech_Command/mel_spect/classifier/"
    os.makedirs(target_dir,exist_ok=True)
    print(target_dir)

    # 设置是否使用cuda
    cuda = "cuda:" + str(args.cuda_id) #指定cuda设备
    use_cuda = not args.no_cuda and torch.cuda.is_available() #判断是否使用cuda
    device = torch.device(cuda if use_cuda else "cpu") #设置设备
    torch.cuda.set_device(args.cuda_id) #设置cuda设备
    kwargs = {'num_workers': args.num_workers, 'pin_memory': False} if use_cuda else {} #设置数据加载的参数 如果使用cuda则设置工作进程数和是否使用pin_memory 为False

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

    # #数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    #使用ImageSet类加载训练集和测试集
    train_set = StudioSet(root='../Speech_Command/audio/train_set_mel_spect_numpy', transform=transform)
    test_set = StudioSet(root='../Speech_Command/audio/test_set_mel_spect_numpy', transform=transform)
    #创建数据加载器，支持批处理，是否打乱数据，多线程加载
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)


    #定义分类Classifier模型和优化器Adam,并设置学习率
    classifier = ResNet18(BasicBlock=BasicBlock,nc=args.nc, nz=args.nz).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True) #最初的学习率

    #初始化最佳分类准确率 和 最佳分类轮次
    best_cl_acc = 0
    best_cl_epoch = 0

    #训练循环
    # Train classifier
    state = {
        'epoch': 0,
        'model': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': 0,
    }
    torch.save(state, target_dir + 'classifier_' + str(0) + '.pth')

    for epoch in range(1, args.epochs + 1):
        #遍历每个训练轮次，调用train函数训练 调用test函数在测试集上评估性能
        train(classifier, args.log_interval, device, train_loader, optimizer, epoch,target_dir)
        cl_acc = test(classifier, device, test_loader,target_dir)

        # #保存中间训练轮次的结果
        # if epoch % 10 == 1:
        #     state = {
        #         'epoch': epoch,
        #         'model': classifier.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'acc': cl_acc,
        #     }
        #     torch.save(state,target_dir+'classifier_'+str(epoch)+'.pth')
        #
        # #保存中间训练轮次的结果
        # if epoch % 25 == 0:
        #     state = {
        #         'epoch': epoch,
        #         'model': classifier.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'acc': cl_acc,
        #     }
        #     torch.save(state,target_dir+'classifier_'+str(epoch)+'.pth')

        # 保存中间训练轮次的结果
        if epoch in [5,10,15,20,25,50,100]:
            state = {
                'epoch': epoch,
                'model': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'acc': cl_acc,
            }
            torch.save(state, target_dir + 'classifier_' + str(epoch) + '.pth')

        #如果当前轮次的准确率高于之前最优值，进行更新
        #更新最佳准确率和最佳轮次
        if cl_acc > best_cl_acc:
            best_cl_acc = cl_acc
            best_cl_epoch = epoch
            state = {
                'epoch': epoch,
                'model': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_cl_acc': best_cl_acc,
            }
            torch.save(state,target_dir+'classifier.pth')

    # #打印最佳模型信息
    # print("Best classifier: epoch {}, acc {:.6f}".format(best_cl_epoch, best_cl_acc))
    # 打印最佳模型信息
    best_info = "Best classifier: epoch {}, acc {:.6f}".format(best_cl_epoch, best_cl_acc)
    print(best_info)

    # 将最佳模型信息写入到文件中
    with open(target_dir + 'best_classifier_info.txt', 'w') as f:
        f.write(best_info + '\n')

if __name__ == '__main__':
    main()