import torch.utils.data
import os.path
import re
import cv2
import xml.etree.ElementTree as ET
import numpy as np

class_name = ('__background__', 'boat', 'building', 'car', 'drone', 'group', 'horseride', 'paraglider',
                'person', 'riding', 'truck', 'wakeboard', 'whale')

class DACDataset(torch.utils.data.Dataset):
    def __init__(self, root, image_set, transform = None):
        """
        Arguments:
            root (string): filepath to DAC2020 dataset
            image_set (string): imageset to use (e.g. "train", "val", "test")
        """
        self.root = root
        self.class_to_index = dict(zip(class_name, range(len(class_name))))
        self.image_set = image_set
        self.transform = transform
        self.ids = []
        for line in open(os.path.join(self.root, "Main", self.image_set + ".txt")):
            self.ids.append(line.rstrip())
    
    def __getitem__(self, index):
        img_id = self.ids[index]
        anno_file = os.path.join(self.root, "Annotations", img_id + ".xml")
        img_file = os.path.join(self.root, "JPEGImages", img_id + ".jpg")

        img = cv2.imread(img_file)
        height, width = img.shape[0], img.shape[1]
        
        bbox = []
        label = []
        anno_object = ET.parse(anno_file).find("object")
        name = anno_object.find("name").text
        main_name = re.sub("[0-9]", "", name)
        label = self.class_to_index[main_name]
        bndbox = anno_object.find("bndbox")
        pts = ['xmin', 'ymin', 'xmax', 'ymax']
        bbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bndbox.find(pt).text) - 1
            cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
            bbox.append(cur_pt)
        bbox = np.array(bbox, dtype=np.float32)
        label = np.array([label])
        if self.transform:
            img, bbox, label = self.transform(img, bbox, label)
        img = img[:, :, (2, 1, 0)] # BGR to RGB
        #print("This is shape of dataset")
        #print(bbox.shape)
        #print(label.shape)
        return img.transpose(2, 0, 1), bbox, label
        new = np.concatenate((bbox, label))
        new = np.concatenate((np.array([img_id]), new))
        # [img, targets]
        return img.transpose(2, 0, 1), 
    
    def __len__(self):
        return len(self.ids)


if __name__ == '__main__':
    root = "/share/DAC2020/dataset"
    train_dataset = DACDataset(root, "train")
    # bboxes = []
    # for i in range(len(train_dataset)):
    #     _, bbox, __ = train_dataset[i]
    #     width = bbox[2] - bbox[0]
    #     height = bbox[3] - bbox[1]
    #     bboxes.append(width * height)
    # bboxes = np.array(bboxes)
    # np.savetxt("object_size.txt", bboxes)
    img, bbox, label = train_dataset[0]
    print(img.shape)
    print(bbox)
    print(label)
