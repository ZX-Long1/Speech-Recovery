import re
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap


def plot_loss_and_accuracy(losses1, losses2, accuracies1, accuracies2, save_dir=None):
    """
    分别绘制两组损失值和两组准确率的图表，并保存为文件（可选）。

    :param losses1: 第一组损失值列表
    :param losses2: 第二组损失值列表
    :param accuracies1: 第一组准确率值列表
    :param accuracies2: 第二组准确率值列表
    :param save_dir: 保存图表的目录路径（可选）
    """
    # # 定义颜色（适合论文用的蓝色）
    # color1 = (0.2980392156862745, 0.4470588235294118, 0.6901960784313725)  # 浅蓝色
    # color2 = (0.3333333333333333, 0.6588235294117647, 0.7803921568627451)  # 稍深一点的蓝色
    color1 = (150 / 255., 165 / 255., 210 / 255.)  # 稍微深一点的蓝色
    color2 = (245 / 255., 145 / 255., 130 / 255.)  # 稍微深一点的橙色


    # 绘制损失值图表
    plt.figure(figsize=(10, 6))  # 调整图像大小
    plt.plot(losses1, label='Mel', color=color1, marker='o', linestyle='-', linewidth=1.5, markersize=4)
    plt.plot(losses2, label='MFCC', color=color2, marker='o', linestyle='-', linewidth=1.5, markersize=4)
    plt.title('Loss over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save_dir:
        loss_save_path = os.path.join(save_dir, 'loss_plot.png')
        plt.savefig(loss_save_path, dpi=800, bbox_inches='tight')
        print(f"Loss chart saved to {loss_save_path}")
    plt.show()

    # 绘制准确率图表
    plt.figure(figsize=(10, 6))  # 调整图像大小
    plt.plot(accuracies1, label='Mel', color=color1, marker='o', linestyle='-', linewidth=1.5, markersize=4)
    plt.plot(accuracies2, label='MFCC', color=color2, marker='o', linestyle='-', linewidth=1.5, markersize=4)
    plt.title('Accuracy over Epochs', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()

    if save_dir:
        acc_save_path = os.path.join(save_dir, 'accuracy_plot.png')
        plt.savefig(acc_save_path, dpi=800, bbox_inches='tight')
        print(f"Accuracy chart saved to {acc_save_path}")
    plt.show()


def extract_loss_and_accuracy(log_file_path):
    # 初始化列表来存储损失值和准确率值
    losses = []
    accuracies = []

    # 正则表达式匹配
    pattern = re.compile(r"Test classifier: Average loss: (\d+\.\d+), Accuracy: \d+/\d+ \((\d+\.\d+)%\)")

    # 打开日志文件并逐行读取
    with open(log_file_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                # 提取损失值和准确率值
                loss = float(match.group(1))
                accuracy = float(match.group(2))
                losses.append(loss)
                accuracies.append(accuracy)

    return losses, accuracies

path1='../SplitLearn/UrbanSound8K/Result/lose_ans_acc/'
os.makedirs(path1, exist_ok=True)

# UrbanSound8K数据集
log_file_path1 = "../SplitLearn/UrbanSound8K/mel_spect/classifier/record_test.txt"  # 替换为你的日志文件路径
log_file_path2 = "../SplitLearn/UrbanSound8K/mfcc/classifier/record_test.txt"
losses1, accuracies1 = extract_loss_and_accuracy(log_file_path1)
losses2, accuracies2 = extract_loss_and_accuracy(log_file_path2)

# 打印提取结果
print("Losses:", losses1)
print("Accuracies:", accuracies1)
# 打印列表的大小
print("Number of losses:", len(losses1))
print("Number of accuracies:", len(accuracies1))

# 打印提取结果
print("Losses:", losses2)
print("Accuracies:", accuracies2)
# 打印列表的大小
print("Number of losses:", len(losses2))
print("Number of accuracies:", len(accuracies2))

# 绘制图表
plot_loss_and_accuracy(losses1,losses2 , accuracies1,accuracies2,path1)


path2='../SplitLearn/Speech_Command/Result/lose_ans_acc/'
os.makedirs(path2, exist_ok=True)

# SpeechCommand数据集
log_file_path1 = "../SplitLearn/Speech_Command/mel_spect/classifier/record_test.txt"  # 替换为你的日志文件路径
log_file_path2 = "../SplitLearn/Speech_Command/mfcc/classifier/record_test.txt"
losses1, accuracies1 = extract_loss_and_accuracy(log_file_path1)
losses2, accuracies2 = extract_loss_and_accuracy(log_file_path2)

# 打印提取结果
print("Losses:", losses1)
print("Accuracies:", accuracies1)
# 打印列表的大小
print("Number of losses:", len(losses1))
print("Number of accuracies:", len(accuracies1))

# 打印提取结果
print("Losses:", losses2)
print("Accuracies:", accuracies2)
# 打印列表的大小
print("Number of losses:", len(losses2))
print("Number of accuracies:", len(accuracies2))

# 绘制图表
plot_loss_and_accuracy(losses1,losses2 , accuracies1,accuracies2,path2)