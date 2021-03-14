import torch
from torch.utils.data import DataLoader

import numpy as np

from dataset import DACDataset
from augmentation import Augmentation, BaseTransform

root = "/share/DAC2020/dataset/"

def get_mean_std(dataset, ratio=0.01):
    """Get mean and std by sample ratio
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=int(len(dataset)*ratio), 
                                             shuffle=True, num_workers=20)
    train = iter(dataloader).next()[0] 
    mean = np.mean(train.numpy(), axis=(0,2,3))
    std = np.std(train.numpy(), axis=(0,2,3))
    return mean, std

train_dataset = DACDataset(root, "train", BaseTransform(320, 160)) # BaseTransform has no Normalize

train_mean, train_std = get_mean_std(train_dataset, 1)

print(train_mean, train_std)
