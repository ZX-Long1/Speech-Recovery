import os
import shutil
import random

# source_folders1 = '../Speech_Command/audio/test_set_mel_spect_numpy'
# target_folders1 = '../Speech_Command/audio/mel_spect_acc_100'
# os.makedirs(target_folders1,exist_ok=True)
#
# # 获取所有文件的路径
# all_files = [os.path.join(source_folders1, filename) for filename in os.listdir(source_folders1)]
# # 打乱文件列表
# random.shuffle(all_files)
#
# t = 0  # 用于计数
# for file_path in all_files:
#     t = t + 1  # 计数
#     if t > 100:
#         break
#     shutil.copy2(file_path, target_folders1)



source_folders2 = '../Speech_Command/audio/test_set_mfcc_numpy'
target_folders2 = '../Speech_Command/audio/mfcc_acc_100'
os.makedirs(target_folders2,exist_ok=True)

# 获取所有文件的路径
all_files = [os.path.join(source_folders2, filename) for filename in os.listdir(source_folders2)]
# 打乱文件列表
random.shuffle(all_files)

t = 0  # 用于计数
for file_path in all_files:
    t = t + 1  # 计数
    if t > 100:
        break
    shutil.copy2(file_path, target_folders2)

