from dataset import DACDataset
from augmentation import Augmentation, BaseTransform

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import random

import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("index", type=int)
# args = parser.parse_args()

root = "/Share/DAC2020/dataset/"

# dataset = DACDataset(root, "train", Augmentation(320, 160))
dataset = DACDataset(root, "train", BaseTransform(320, 160))

# img, bbox, label = dataset[args.index]
img, bbox, label = dataset[random.randint(0, 56000)]

img = np.transpose(img, (1, 2, 0))
img = img.astype(int)
img += (113, 116, 105)

x, y = bbox[0]*320, bbox[1]*160
x, y = x.astype(int), y.astype(int)
width = (bbox[2] - bbox[0])*320
height = (bbox[3] - bbox[1])*160
width, height = width.astype(int), height.astype(int)

plt.imshow(img)
rect = patches.Rectangle((x, y), width, height, linewidth=1,edgecolor='r',facecolor='none')

cur_axis = plt.gca()
cur_axis.add_patch(rect)

plt.savefig("test.jpg")

print(x, y, width, height)

class_name = ('__background__', 'boat', 'building', 'car', 'drone', 'group', 'horseride', 'paraglider',
                'person', 'riding', 'truck', 'wakeboard', 'whale')
print(class_name[label])