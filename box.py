# import matplotlib.pyplot as plt
# import cv2, utils
# import torchvision.transforms.functional as F
# import torch.nn.functional as FF
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
#
# mode = "BlackBox"
#
# # 这一步记录每个分割点/噪声/丢失率对应的生成数据 与原始数据的mse，
# def box():
#     noise = [1, 3, 5, 10, 50] #这里是之前方案里用到的噪声用来命名每个噪声对应的文件夹
#     sp = 4
#     data_ssim = []
#     data_mse = [] # 存放每个epoch的数据
#
#     for i in noise:
#         # 4个分割点
#         noise_epoch_ssim = []
#         noise_epoch_mse = []
#         # print(i)
#         for j in range(1, 4, 2):
#             layer_record_mse = []
#             for k in range(1, 51):
#                 image1 = cv2.imread(
#                     'face/' + 'noisel/' + mode + '/' + 'noise' + str(i) + '/sp' + str(j) + '/' + str(
#                         k) + '.png', 0)  # 需要进行计算mse的a文件夹里的数据
#                 image2 = cv2.imread('face/Origin/' + str(k) + '.png', 0)  #  需要进行计算mse的a文件夹里的数据
#                 img1 = F.to_tensor(image1)
#                 img2 = F.to_tensor(image2)
#                 img1 = img1.unsqueeze(0)
#                 img2 = img2.unsqueeze(0)
#                 temp = FF.mse_loss(img1, img2)
#                 layer_record_mse.append(temp.item())
#             noise_epoch_mse.append(layer_record_mse)
#         data_mse.append(noise_epoch_mse) # 这里记录了五组数据
#
#
#     labels = ["$Split1$", "$Split3$"] # 图例， 比如噪声我们中选取了两个分割点 split1和split3；如果是画mse
#     colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]
#     # 两个箱型图的颜色 RGB （均为0~1的数据）
#
#     # 绘制箱型图 5个bplot代表五个横坐标
#     bplot = plt.boxplot(data_mse[0], patch_artist=True, labels=labels, positions=(1, 1.4), widths=0.29,
#                          flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
#
#     for patch, color in zip(bplot['boxes'], colors):
#         patch.set_facecolor(color)
#     for median in bplot['medians']:
#         median.set(color='black')#, linewidth=3
#
#     bplot2 = plt.boxplot(data_mse[1], patch_artist=True, labels=labels, positions=(2.1, 2.5), widths=0.29,
#                          flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
#
#     for patch, color in zip(bplot2['boxes'], colors):
#         patch.set_facecolor(color)
#     for median in bplot2['medians']:
#         median.set(color='black')  # , linewidth=3
#
#     bplot3 = plt.boxplot(data_mse[2], patch_artist=True, labels=labels, positions=(3.2, 3.6), widths=0.29,
#                          flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
#
#     for patch, color in zip(bplot3['boxes'], colors):
#         patch.set_facecolor(color)
#     for median in bplot3['medians']:
#         median.set(color='black')#, linewidth=3
#
#     bplot4 = plt.boxplot(data_mse[3], patch_artist=True, labels=labels, positions=(4.3, 4.7), widths=0.29,
#                          flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
#
#     for patch, color in zip(bplot4['boxes'], colors):
#         patch.set_facecolor(color)
#     for median in bplot4['medians']:
#         median.set(color='black')#, linewidth=3
#
#     bplot5 = plt.boxplot(data_mse[4], patch_artist=True, labels=labels, positions=(5.4, 5.8), widths=0.29,
#                          flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
#
#     for patch, color in zip(bplot5['boxes'], colors):
#         patch.set_facecolor(color)
#     for median in bplot5['medians']:
#         median.set(color='black')
#
#
#
#     x_position = [1.2, 2.3, 3.4, 4.5, 5.6] #每一组两个箱型图中间的坐标
#     x_position_fmt = ["$\sigma=0.2$", "$\sigma=0.4$", "$\sigma=0.6$", "$\sigma=0.8$", "$\sigma=1.0$"] # 横坐标命名
#
#     plt.xticks(x_position, x_position_fmt, fontsize = 16)
#     plt.yticks(fontsize=16)
#
#     plt.ylabel('MSE', fontsize = 18)
#     plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
#     legend = plt.legend(bplot['boxes'], labels, loc='upper left', title='Split Point', fontsize= 12) #, fontsize = 12) # 绘制表示框，右下角绘制
#     legend.get_title().set_fontsize('12')
#     plt.savefig('noisel_face_b_mse.pdf', bbox_inches='tight')
#     plt.tight_layout()
#     # plt.savefig(fname="pic.png", figsize=[10, 10])
#     plt.show()
#
#
# box()


import matplotlib.pyplot as plt
import utils
import torchvision.transforms.functional as F
import torch.nn.functional as FF
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def noise_mse_box():
    noise = [1,5,10,15,20] #这里是之前方案里用到的噪声用来命名每个噪声对应的文件夹
    noise_mse_dir='../mse_plot/'
    split1_mse=np.load(noise_mse_dir+'noise_split1.npy')
    split3_mse=np.load(noise_mse_dir+'noise_split3.npy')

    labels = ["$Split1$", "$Split3$"] # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]# 两个箱型图的颜色 RGB （均为0~1的数据）

    # # 设置画布尺寸，增大宽度，使得图形更宽松
    # plt.figure(figsize=(8, 4.8))  # 修改这里，宽度设置为12，高度设置为6，您可以根据需要调整

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([split1_mse[1],split3_mse[1]], patch_artist=True, labels=labels, positions=(1, 1.4), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')#, linewidth=3

    bplot2 = plt.boxplot([split1_mse[2],split3_mse[2]] ,patch_artist=True, labels=labels, positions=(2.1, 2.5), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([split1_mse[3],split3_mse[3]], patch_artist=True, labels=labels, positions=(3.2, 3.6), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')#, linewidth=3

    bplot4 = plt.boxplot([split1_mse[4],split3_mse[4]], patch_artist=True, labels=labels, positions=(4.3, 4.7), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')#, linewidth=3

    bplot5 = plt.boxplot([split1_mse[5],split3_mse[5]], patch_artist=True, labels=labels, positions=(5.4, 5.8), widths=0.29,flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot5['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot5['medians']:
        median.set(color='black')

    x_position = [1.2, 2.3, 3.4, 4.5, 5.6] #每一组两个箱型图中间的坐标
    x_position_fmt = ["$\sigma=0.1$", "$\sigma=0.5$", "$\sigma=1.0$", "$\sigma=1.5$","$\sigma=2.0$"] # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize = 14)
    plt.yticks(fontsize=16)

    plt.ylabel('MSE', fontsize = 18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper left', title='Split Point', fontsize= 12) #, fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(noise_mse_dir + 'noise.png', dpi=600,bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def dropout_mse_box():
    dropout = [1,2,3,4,5]
    mse_dir='../mse_plot/'
    split1_mse=np.load(mse_dir+'dropout_split1.npy')
    split3_mse=np.load(mse_dir+'dropout_split3.npy')

    labels = ["$Split1$", "$Split3$"] # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]# 两个箱型图的颜色 RGB （均为0~1的数据）

    # # 设置画布尺寸，增大宽度，使得图形更宽松
    # plt.figure(figsize=(8, 4.8))  # 修改这里，宽度设置为12，高度设置为6，您可以根据需要调整

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([split1_mse[1],split3_mse[1]], patch_artist=True, labels=labels, positions=(1, 1.4), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')#, linewidth=3

    bplot2 = plt.boxplot([split1_mse[2],split3_mse[2]] ,patch_artist=True, labels=labels, positions=(2.1, 2.5), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([split1_mse[3],split3_mse[3]], patch_artist=True, labels=labels, positions=(3.2, 3.6), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')#, linewidth=3

    bplot4 = plt.boxplot([split1_mse[4],split3_mse[4]], patch_artist=True, labels=labels, positions=(4.3, 4.7), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')#, linewidth=3

    bplot5 = plt.boxplot([split1_mse[5],split3_mse[5]], patch_artist=True, labels=labels, positions=(5.4, 5.8), widths=0.29,flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot5['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot5['medians']:
        median.set(color='black')


    x_position = [1.2, 2.3, 3.4, 4.5, 5.6] #每一组两个箱型图中间的坐标
    x_position_fmt = ["$r=0.1$", "$r=0.2$", "$r=0.3$", "$r=0.4$","$r=0.5$"] # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize = 14)
    plt.yticks(fontsize=16)

    plt.ylabel('MSE', fontsize = 18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper left', title='Split Point', fontsize= 12) #, fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(mse_dir + 'dropout.png', dpi=600,bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def whitebox_melspect_mse_box():
    mse_dir = '../mse_plot/'
    mse1= np.load(mse_dir + 'UrbanSound8K_whitebox_melspect.npy')
    mse2 = np.load(mse_dir + 'SpeechCommand_whitebox_melspect.npy')

    labels = ["$UrbanSound8K$", "$Google speech commands$"]  # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]  # 两个箱型图的颜色 RGB （均为0~1的数据）

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([mse1[0], mse2[0]], patch_artist=True, labels=labels, positions=(1, 1.4),
                        widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')  # , linewidth=3

    bplot2 = plt.boxplot([mse1[1], mse2[1]], patch_artist=True, labels=labels, positions=(2.1, 2.5),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([mse1[2], mse2[2]], patch_artist=True, labels=labels, positions=(3.2, 3.6),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')  # , linewidth=3

    bplot4 = plt.boxplot([mse1[3], mse2[3]], patch_artist=True, labels=labels, positions=(4.3, 4.7),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')  # , linewidth=3


    x_position = [1.2, 2.3, 3.4, 4.5]  # 每一组两个箱型图中间的坐标
    x_position_fmt = ["$Split1$", "$Split2$", "$Split3$", "$Split4$"]  # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylabel('MSE', fontsize=18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper left', title='Dataset',
                        fontsize=12)  # , fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(mse_dir + 'whitebox_melspect.png', dpi=600, bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def whitebox_mfcc_mse_box():
    mse_dir = '../mse_plot/'
    mse1 = np.load(mse_dir + 'UrbanSound8K_whitebox_mfcc.npy')
    mse2 = np.load(mse_dir + 'SpeechCommand_whitebox_mfcc.npy')

    labels = ["$UrbanSound8K$", "$Google speech commands$"]  # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]  # 两个箱型图的颜色 RGB （均为0~1的数据）

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([mse1[0], mse2[0]], patch_artist=True, labels=labels, positions=(1, 1.4),
                        widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')  # , linewidth=3

    bplot2 = plt.boxplot([mse1[1], mse2[1]], patch_artist=True, labels=labels, positions=(2.1, 2.5),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([mse1[2], mse2[2]], patch_artist=True, labels=labels, positions=(3.2, 3.6),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')  # , linewidth=3

    bplot4 = plt.boxplot([mse1[3], mse2[3]], patch_artist=True, labels=labels, positions=(4.3, 4.7),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')  # , linewidth=3

    x_position = [1.2, 2.3, 3.4, 4.5]  # 每一组两个箱型图中间的坐标
    x_position_fmt = ["$Split1$", "$Split2$", "$Split3$", "$Split4$"]  # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylabel('MSE', fontsize=18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper left', title='Dataset',
                        fontsize=12)  # , fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(mse_dir + 'whitebox_mfcc.png', dpi=600, bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def blackbox_melspect_mse_box():
    mse_dir = '../mse_plot/'
    mse1 = np.load(mse_dir + 'UrbanSound8K_blackbox_melspect.npy')
    mse2 = np.load(mse_dir + 'SpeechCommand_blackbox_melspect.npy')

    labels = ["$UrbanSound8K$", "$Google speech commands$"]  # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]  # 两个箱型图的颜色 RGB （均为0~1的数据）

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([mse1[0], mse2[0]], patch_artist=True, labels=labels, positions=(1, 1.4),
                        widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')  # , linewidth=3

    bplot2 = plt.boxplot([mse1[1], mse2[1]], patch_artist=True, labels=labels, positions=(2.1, 2.5),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([mse1[2], mse2[2]], patch_artist=True, labels=labels, positions=(3.2, 3.6),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')  # , linewidth=3

    bplot4 = plt.boxplot([mse1[3], mse2[3]], patch_artist=True, labels=labels, positions=(4.3, 4.7),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')  # , linewidth=3

    x_position = [1.2, 2.3, 3.4, 4.5]  # 每一组两个箱型图中间的坐标
    x_position_fmt = ["$Split1$", "$Split2$", "$Split3$", "$Split4$"]  # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylabel('MSE', fontsize=18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper left', title='Dataset',
                        fontsize=12)  # , fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(mse_dir + 'blackbox_melspect.png', dpi=600, bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def blackbox_mfcc_mse_box():
    mse_dir = '../mse_plot/'
    mse1 = np.load(mse_dir + 'UrbanSound8K_blackbox_mfcc.npy')
    mse2 = np.load(mse_dir + 'SpeechCommand_blackbox_mfcc.npy')

    labels = ["$UrbanSound8K$", "$Google speech commands$"]  # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]  # 两个箱型图的颜色 RGB （均为0~1的数据）

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([mse1[0], mse2[0]], patch_artist=True, labels=labels, positions=(1, 1.4),
                        widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')  # , linewidth=3

    bplot2 = plt.boxplot([mse1[1], mse2[1]], patch_artist=True, labels=labels, positions=(2.1, 2.5),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([mse1[2], mse2[2]], patch_artist=True, labels=labels, positions=(3.2, 3.6),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')  # , linewidth=3

    bplot4 = plt.boxplot([mse1[3], mse2[3]], patch_artist=True, labels=labels, positions=(4.3, 4.7),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')  # , linewidth=3

    x_position = [1.2, 2.3, 3.4, 4.5]  # 每一组两个箱型图中间的坐标
    x_position_fmt = ["$Split1$", "$Split2$", "$Split3$", "$Split4$"]  # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylabel('MSE', fontsize=18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper left', title='Dataset',
                        fontsize=12)  # , fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(mse_dir + 'blackbox_mfcc.png', dpi=600, bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def noise_snr_box():
    noise = [1, 5,10,15,20] #这里是之前方案里用到的噪声用来命名每个噪声对应的文件夹
    noise_mse_dir='../snr_plot/'
    split1_mse=np.load(noise_mse_dir+'noise_split1.npy')
    split3_mse=np.load(noise_mse_dir+'noise_split3.npy')

    labels = ["$Split1$", "$Split3$"] # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]# 两个箱型图的颜色 RGB （均为0~1的数据）

    # # 设置画布尺寸，增大宽度，使得图形更宽松
    # plt.figure(figsize=(8, 4.8))  # 修改这里，宽度设置为12，高度设置为6，您可以根据需要调整

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([split1_mse[1],split3_mse[1]], patch_artist=True, labels=labels, positions=(1, 1.4), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')#, linewidth=3

    bplot2 = plt.boxplot([split1_mse[2],split3_mse[2]] ,patch_artist=True, labels=labels, positions=(2.1, 2.5), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([split1_mse[3],split3_mse[3]], patch_artist=True, labels=labels, positions=(3.2, 3.6), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')#, linewidth=3

    bplot4 = plt.boxplot([split1_mse[4],split3_mse[4]], patch_artist=True, labels=labels, positions=(4.3, 4.7), widths=0.29,flierprops = dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')#, linewidth=3

    bplot5 = plt.boxplot([split1_mse[5],split3_mse[5]], patch_artist=True, labels=labels, positions=(5.4, 5.8), widths=0.29,flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot5['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot5['medians']:
        median.set(color='black')

    x_position = [1.2, 2.3, 3.4, 4.5, 5.6] #每一组两个箱型图中间的坐标
    x_position_fmt = ["$\sigma=0.1$", "$\sigma=0.5$", "$\sigma=1.0$", "$\sigma=1.5$","$\sigma=2.0$"] # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize = 14)
    plt.yticks(fontsize=16)

    plt.ylabel('SNR', fontsize = 18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper right', title='Split Point', fontsize= 12) #, fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(noise_mse_dir + 'noise.png', dpi=600,bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def dropout_snr_box():
    dropout = [1,2,3,4,5]
    mse_dir='../snr_plot/'
    split1_mse=np.load(mse_dir+'dropout_split1.npy')
    split3_mse=np.load(mse_dir+'dropout_split3.npy')

    labels = ["$Split1$", "$Split3$"] # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]# 两个箱型图的颜色 RGB （均为0~1的数据）

    # # 设置画布尺寸，增大宽度，使得图形更宽松
    # plt.figure(figsize=(8, 4.8))  # 修改这里，宽度设置为12，高度设置为6，您可以根据需要调整

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([split1_mse[1], split3_mse[1]], patch_artist=True, labels=labels, positions=(1, 1.4),
                        widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')  # , linewidth=3

    bplot2 = plt.boxplot([split1_mse[2], split3_mse[2]], patch_artist=True, labels=labels, positions=(2.1, 2.5),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([split1_mse[3], split3_mse[3]], patch_artist=True, labels=labels, positions=(3.2, 3.6),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')  # , linewidth=3

    bplot4 = plt.boxplot([split1_mse[4], split3_mse[4]], patch_artist=True, labels=labels, positions=(4.3, 4.7),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')  # , linewidth=3

    bplot5 = plt.boxplot([split1_mse[5], split3_mse[5]], patch_artist=True, labels=labels, positions=(5.4, 5.8),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot5['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot5['medians']:
        median.set(color='black')

    x_position = [1.2, 2.3, 3.4, 4.5, 5.6] #每一组两个箱型图中间的坐标
    x_position_fmt = ["$r=0.1$", "$r=0.2$", "$r=0.3$", "$r=0.4$","$r=0.5$"] # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize = 14)
    plt.yticks(fontsize=16)

    plt.ylabel('SNR', fontsize = 18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper right', title='Split Point', fontsize= 12) #, fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(mse_dir + 'dropout.png', dpi=600,bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def whitebox_melspect_snr_box():
    mse_dir = '../snr_plot/'
    mse1= np.load(mse_dir + 'UrbanSound8K_whitebox_melspect.npy')
    mse2 = np.load(mse_dir + 'SpeechCommand_whitebox_melspect.npy')

    labels = ["$UrbanSound8K$", "$Google speech commands$"]  # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]  # 两个箱型图的颜色 RGB （均为0~1的数据）

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([mse1[0], mse2[0]], patch_artist=True, labels=labels, positions=(1, 1.4),
                        widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')  # , linewidth=3

    bplot2 = plt.boxplot([mse1[1], mse2[1]], patch_artist=True, labels=labels, positions=(2.1, 2.5),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([mse1[2], mse2[2]], patch_artist=True, labels=labels, positions=(3.2, 3.6),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')  # , linewidth=3

    bplot4 = plt.boxplot([mse1[3], mse2[3]], patch_artist=True, labels=labels, positions=(4.3, 4.7),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')  # , linewidth=3


    x_position = [1.2, 2.3, 3.4, 4.5]  # 每一组两个箱型图中间的坐标
    x_position_fmt = ["$Split1$", "$Split2$", "$Split3$", "$Split4$"]  # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylabel('SNR', fontsize=18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper right', title='Dataset',
                        fontsize=12)  # , fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(mse_dir + 'whitebox_melspect.png', dpi=600, bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def whitebox_mfcc_snr_box():
    mse_dir = '../snr_plot/'
    mse1 = np.load(mse_dir + 'UrbanSound8K_whitebox_mfcc.npy')
    mse2 = np.load(mse_dir + 'SpeechCommand_whitebox_mfcc.npy')

    labels = ["$UrbanSound8K$", "$Google speech commands$"]  # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]  # 两个箱型图的颜色 RGB （均为0~1的数据）

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([mse1[0], mse2[0]], patch_artist=True, labels=labels, positions=(1, 1.4),
                        widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')  # , linewidth=3

    bplot2 = plt.boxplot([mse1[1], mse2[1]], patch_artist=True, labels=labels, positions=(2.1, 2.5),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([mse1[2], mse2[2]], patch_artist=True, labels=labels, positions=(3.2, 3.6),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')  # , linewidth=3

    bplot4 = plt.boxplot([mse1[3], mse2[3]], patch_artist=True, labels=labels, positions=(4.3, 4.7),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')  # , linewidth=3

    x_position = [1.2, 2.3, 3.4, 4.5]  # 每一组两个箱型图中间的坐标
    x_position_fmt = ["$Split1$", "$Split2$", "$Split3$", "$Split4$"]  # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylabel('SNR', fontsize=18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper right', title='Dataset',
                        fontsize=12)  # , fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(mse_dir + 'whitebox_mfcc.png', dpi=600, bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def blackbox_melspect_snr_box():
    mse_dir = '../snr_plot/'
    mse1 = np.load(mse_dir + 'UrbanSound8K_blackbox_melspect.npy')
    mse2 = np.load(mse_dir + 'SpeechCommand_blackbox_melspect.npy')

    labels = ["$UrbanSound8K$", "$Google speech commands$"]  # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]  # 两个箱型图的颜色 RGB （均为0~1的数据）

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([mse1[0], mse2[0]], patch_artist=True, labels=labels, positions=(1, 1.4),
                        widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')  # , linewidth=3

    bplot2 = plt.boxplot([mse1[1], mse2[1]], patch_artist=True, labels=labels, positions=(2.1, 2.5),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([mse1[2], mse2[2]], patch_artist=True, labels=labels, positions=(3.2, 3.6),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')  # , linewidth=3

    bplot4 = plt.boxplot([mse1[3], mse2[3]], patch_artist=True, labels=labels, positions=(4.3, 4.7),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')  # , linewidth=3

    x_position = [1.2, 2.3, 3.4, 4.5]  # 每一组两个箱型图中间的坐标
    x_position_fmt = ["$Split1$", "$Split2$", "$Split3$", "$Split4$"]  # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylabel('SNR', fontsize=18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper right', title='Dataset',
                        fontsize=12)  # , fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(mse_dir + 'blackbox_melspect.png', dpi=600, bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def blackbox_mfcc_snr_box():
    mse_dir = '../snr_plot/'
    mse1 = np.load(mse_dir + 'UrbanSound8K_blackbox_mfcc.npy')
    mse2 = np.load(mse_dir + 'SpeechCommand_blackbox_mfcc.npy')

    labels = ["$UrbanSound8K$", "$Google speech commands$"]  # 图例， 噪声中选取了两个分割点 split1和split3；如果是画mse
    colors = [(161 / 255., 169 / 255., 208 / 255.), (240 / 255., 152 / 255., 140 / 255.)]  # 两个箱型图的颜色 RGB （均为0~1的数据）

    # 绘制箱型图 5个bplot代表五个横坐标
    bplot = plt.boxplot([mse1[0], mse2[0]], patch_artist=True, labels=labels, positions=(1, 1.4),
                        widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot['medians']:
        median.set(color='black')  # , linewidth=3

    bplot2 = plt.boxplot([mse1[1], mse2[1]], patch_artist=True, labels=labels, positions=(2.1, 2.5),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set(color='black')  # , linewidth=3

    bplot3 = plt.boxplot([mse1[2], mse2[2]], patch_artist=True, labels=labels, positions=(3.2, 3.6),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set(color='black')  # , linewidth=3

    bplot4 = plt.boxplot([mse1[3], mse2[3]], patch_artist=True, labels=labels, positions=(4.3, 4.7),
                         widths=0.29, flierprops=dict(marker="+", markerfacecolor='black', markersize=4))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set(color='black')  # , linewidth=3

    x_position = [1.2, 2.3, 3.4, 4.5]  # 每一组两个箱型图中间的坐标
    x_position_fmt = ["$Split1$", "$Split2$", "$Split3$", "$Split4$"]  # 横坐标命名

    plt.xticks(x_position, x_position_fmt, fontsize=16)
    plt.yticks(fontsize=16)

    plt.ylabel('SNR', fontsize=18)
    plt.grid(linestyle="--", alpha=0.3)  # 绘制图中虚线 透明度0.3
    legend = plt.legend(bplot['boxes'], labels, loc='upper right', title='Dataset',
                        fontsize=12)  # , fontsize = 12) # 绘制表示框，右下角绘制
    legend.get_title().set_fontsize('12')
    plt.tight_layout()
    # plt.savefig(fname="pic.png", figsize=[10, 10])

    plt.savefig(mse_dir + 'blackbox_mfcc.png', dpi=600, bbox_inches='tight')  # 保存为高质量图片
    plt.show()

def main():
    # noise_mse_box()
    # dropout_mse_box()
    # whitebox_melspect_mse_box()
    # whitebox_mfcc_mse_box()
    # blackbox_melspect_mse_box()
    # blackbox_mfcc_mse_box()

    # noise_snr_box()
    # dropout_snr_box()
    # whitebox_melspect_snr_box()
    # whitebox_mfcc_snr_box()
    # blackbox_melspect_snr_box()
    # blackbox_mfcc_snr_box()

if __name__ == '__main__':
    main()
