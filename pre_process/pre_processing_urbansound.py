import glob
import shutil
import librosa
import librosa.display
import numpy as np
import pandas as pd
import skimage
from matplotlib import cm
from matplotlib import pyplot as plt
from utils import mfcc_image, save_raw_vectors, spectrogram_image,load_audio_file

if __name__ == "__main__":
	hop_length = 512 
	n_mels = 128 
	time_steps = 384

	df = pd.read_csv("../../UrbanSound8K/UrbanSound8K/metadata/UrbanSound8K.csv")
	print(len(df))

	#遍历csv数据说明文件
	for i in range(len(df)):
		print(i)
		current_row = df.iloc[i]
		# print(current_row)
		current_file_path = "../../UrbanSound8K/UrbanSound8K/audio/fold"+str(current_row["fold"])+"/"+str(current_row["slice_file_name"])
		# print(current_file_path)
		file_id = current_row["slice_file_name"].split(".")[0]   #根据键值获取对应文件名，然后以'.'作为分隔符，然后取分隔结果的第一部分
		# print(file_id)

		mfcc_save_path = "../../UrbanSound8K/UrbanSound8K/mfcc/"+str(file_id)+".png"
		spec_save_path = "../../UrbanSound8K/UrbanSound8K/spec/"+str(file_id)+".png"
		raw_save_path = "../../UrbanSound8K/UrbanSound8K/raw_vectors/"+str(file_id)+".npy"
		wav_save_path = "../../UrbanSound8K/UrbanSound8K/wav_files/"+str(file_id)+".wav"
		# print(mfcc_save_path)
		# print(spec_save_path)
		# print(raw_save_path)
		# print(wav_save_path)

		y = load_audio_file(current_file_path)
		sr = 22050
		# print(y.shape)

		mfcc_image(y,sr,mfcc_save_path)
		spectrogram_image(y, sr=sr, hop_length=hop_length, n_mels=n_mels, save_path = spec_save_path)
		save_raw_vectors(y,raw_save_path)
		shutil.copyfile(current_file_path,wav_save_path)
