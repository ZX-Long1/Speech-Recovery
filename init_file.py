import os
import shutil
import numpy as np
# 源文件夹路径列表
source_folders = '..\\UrbanSound8K\\UrbanSound8K\\audio\\'
# 目标文件夹的根目录
target_folders = '..\\UrbanSound8K\\UrbanSound8K\\audio\\class'
#记录重新写入的11个文件夹中文件的数量
cnt=[0]*10
#遍历fold1-10
for i in range(1,11):
    now_source_folders=source_folders+'fold'+str(i) #访问fold-i文件夹
    #os.listdir()以路径作为参数，返回一个包含路径下所有文件和子目录名称的列表
    for filename in os.listdir(now_source_folders):
        #os.path.join 用于进行路径组合
        file_path=os.path.join(now_source_folders,filename)
        #split函数是一个字符串，根据指定的分隔符将字符串分割成多个部分，返回列表
        split_filename=filename.split('-')
        idd=100
        if(len(split_filename)>1):
            idd=int(split_filename[1])
        else:
            continue
        if(idd<0 or idd>9):
            print(target_folders+str(idd))
        # 首先判断目标文件夹是否存在，不存在则创建新文件夹
        if not os.path.exists(target_folders+str(idd)):
            try:
                os.makedirs(target_folders+str(idd))
            except OSError as e:
                pass

        #统计不同类别的文件数量
        cnt[idd]=cnt[idd]+1

        #根据类别进行文件的复制
        shutil.copy2(file_path, target_folders+str(idd))

print(cnt)

# 首先判断目标文件夹是否存在，不存在则创建新文件夹  train_set  class_set
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

#遍历class0-9
for i in range(0,10):
    now_source_folders=source_folders+'class'+str(i) #class-i文件夹
    print(now_source_folders)
    t = 0 #用于计数
    for filename in os.listdir(now_source_folders):
        file_path = os.path.join(now_source_folders, filename)
        t=t+1#计数
        #训练集取原数据的20%，测试集取原数据的80%
        if t<cnt[i]/5:
            shutil.copy2(file_path, source_folders + 'test_set')
        else:
            shutil.copy2(file_path, source_folders + 'train_set')

