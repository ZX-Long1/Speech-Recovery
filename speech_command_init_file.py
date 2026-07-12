import os
import shutil
import random
import numpy as np

# 源文件夹路径列表
source_folders = '..\\Speech_Command\\speech_commands\\'
# 目标文件夹的根目录
target_folders = '..\\Speech_Command\\audio\\'

classname=['backward','bed','bird','cat','dog','down','eight','five','follow','forward','four','go','happy','house','learn',
           'left','marvin','nine','no','off','on','one','right','seven','sheila','six','stop','three','tree','two','up',
           'visual','wow','yes','zero']
# print(len(classname))

# #记录每个类别音频的数量
cnt=[0]*35
#遍历所有类别文件夹
for i in range(0,len(classname)):
    now_source_folders=source_folders+classname[i]
    # os.listdir()以路径作为参数，返回一个包含路径下所有文件和子目录名称的列表
    for filename in os.listdir(now_source_folders):
        cnt[i]+=1
print(cnt)
print(sum(cnt))

# 首先判断目标文件夹是否存在，不存在则创建新文件夹  train_set  test_set
if not os.path.exists(target_folders + 'train_set'):
    try:
        os.makedirs(target_folders + 'train_set')
    except OSError as e:
        pass

if not os.path.exists(target_folders + 'test_set'):
    try:
        os.makedirs(target_folders + 'test_set')
    except OSError as e:
        pass

#遍历所有类别文件夹
for i in range(0,len(classname)):
    now_source_folders=source_folders+classname[i]
    print(i+1,' ',now_source_folders)

    # 获取所有文件的路径
    all_files = [os.path.join(now_source_folders, filename) for filename in os.listdir(now_source_folders)]
    # 打乱文件列表
    random.shuffle(all_files)

    t = 0  # 用于计数
    for file_path in all_files:
        t = t + 1  # 计数
        # 训练集取原数据的80%，测试集取原数据的20%
        if t <= cnt[i] / 5:
            shutil.copy2(file_path, target_folders + 'test_set'+'//'+str(i)+'_'+classname[i]+str(t)+'.wav')
        else:
            shutil.copy2(file_path, target_folders + 'train_set'+'//'+str(i)+'_'+classname[i]+str(t)+'.wav')



