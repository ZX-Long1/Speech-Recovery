import os, json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class CremaDataset(Dataset):
    def __init__(self, root, feat_type='mel', transform=None, norm_info=None):
        self.root = root
        self.feat_type = feat_type
        self.transform = transform

        if norm_info is None:
            norm_info = json.load(open(f'{os.path.dirname(root)}/norm_info.json'))
        self.min_val = norm_info[f'{feat_type}_min']
        self.max_val = norm_info[f'{feat_type}_max']

        self.files = sorted([f for f in os.listdir(root) if f.endswith(f'_{feat_type}.npy')])
        self.data, self.labels = [], []
        for fname in self.files:
            arr = np.load(os.path.join(root, fname), allow_pickle=True)
            self.data.append(arr)
            emo = fname.split('_')[2]
            label = {'ANG': 0, 'DIS': 1, 'FEA': 2, 'HAP': 3, 'NEU': 4, 'SAD': 5}[emo]
            self.labels.append(label)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        np.random.seed(66)
        perm = np.arange(len(self.data))
        np.random.shuffle(perm)
        self.data = self.data[perm]
        self.labels = self.labels[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feat = self.data[idx]
        label = self.labels[idx]
        norm = ((feat - self.min_val) / (self.max_val - self.min_val)) * 255
        norm = np.uint8(norm)
        img = Image.fromarray(norm)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
