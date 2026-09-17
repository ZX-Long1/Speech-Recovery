import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ManifestDataset(Dataset):
    def __init__(self, root, manifest, min_val, max_val, label_fn,
                 transform=None, seed=66, limit_per_class=None, n_classes=None):
        self.root = root
        self.min_val = min_val
        self.max_val = max_val
        self.label_fn = label_fn
        self.transform = transform
        items = [l.strip() for l in open(manifest) if l.strip()]
        if limit_per_class is not None:
            seen = {}
            picked = []
            for it in items:
                c = label_fn(os.path.basename(it).replace('_mel.npy', ''))
                if seen.get(c, 0) < limit_per_class:
                    seen[c] = seen.get(c, 0) + 1
                    picked.append(it)
            items = picked
        rng = np.random.RandomState(seed)
        perm = np.arange(len(items))
        rng.shuffle(perm)
        self.items = [items[i] for i in perm]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel = self.items[idx]
        path = os.path.join(self.root, rel)
        feat = np.load(path)
        norm = ((feat - self.min_val) / (self.max_val - self.min_val)) * 255
        img = np.uint8(norm)
        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        label = self.label_fn(os.path.basename(rel).replace('_mel.npy', ''))
        return img, label
