# 下载数据
from torchaudio import datasets

datasets.SPEECHCOMMANDS(
    root="../Speech_Command",  # 你保存数据的路径
    url='speech_commands_v0.02',  # 下载数据版本URL
    folder_in_archive='SpeechCommands',
    download=True  # 这个记得选True
)