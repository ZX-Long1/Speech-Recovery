import argparse, json
import datetime
import os
import logging
import torch, random
# from energy_module import analyse, HardwareModel


# -------------------------
# 输出 CUDA 使用信息
# -------------------------
print("\n========== CUDA Information ==========")
if torch.cuda.is_available():
	print(f"CUDA available")
	print(f"CUDA version: {torch.version.cuda}")
	print(f"PyTorch version: {torch.__version__}")
	print(f"GPU count: {torch.cuda.device_count()}")
	for i in range(torch.cuda.device_count()):
		print(f"  [{i}] {torch.cuda.get_device_name(i)}")
	cuda_id=0
	cuda = "cuda:" + str(cuda_id)
	torch.cuda.set_device(cuda)
	print(f"Current device index: {torch.cuda.current_device()}")
	print(f"Current device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print("======================================\n")