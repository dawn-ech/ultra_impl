import torch
import cv2
import numpy as np
from numpy import random


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[2:], box_b[2:])
    min_xy = np.maximum(box_a[:2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[0] * inter[1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[2]-box_a[0]) *
              (box_a[3]-box_a[1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, box=None, label=None):
        for t in self.transforms:
            img, box, label = t(img, box, label)
        return img, box, label


class ConvertFromInts:
    def __call__(self, image, box=None, label=None):
        return image.astype(np.float32), box, label


class SubtractMeans:
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, box=None, label=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), box, label

class Normalize:
    def __init__(self, mean, std):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, image, box=None, label=None):
        image = image.astype(np.float32)
        image -= self.mean
        image /= self.std
        return image.astype(np.float32), box, label

class MinMaxNormalize:
    def __init__(self, max=255, min=0):
        self.max = max
        self.min = min

    def __call__(self, image, box=None, label=None):
        image = image.astype(np.float32)
        image = (image - self.min) / (self.max + 1 - self.min)
        return image.astype(np.float32), box, label


class ToAbsoluteCoords:
    def __call__(self, image, box=None, label=None):
        height, width, channels = image.shape
        box[0] *= width
        box[2] *= width
        box[1] *= height
        box[3] *= height

        return image, box, label


class ToPercentCoords:
    def __call__(self, image, box=None, label=None):
        height, width, channels = image.shape
        box[0] /= width
        box[2] /= width
        box[1] /= height
        box[3] /= height

        return image, box, label


class Resize:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, image, box=None, label=None):
        image = cv2.resize(image, (self.width, self.height))
        return image, box, label


class RandomSaturation:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, box=None, label=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, box, label


class RandomHue:
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, box=None, label=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, box, label


class RandomLightingNoise:
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, box=None, label=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, box, label


class ConvertColor:
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, box=None, label=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, box, label


class RandomContrast:
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, box=None, label=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, box, label


class RandomBrightness:
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, box=None, label=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, box, label


class RandomSampleCrop:
    """Crop
    Arguments:
        img (Image): the image being input during training
        box (Tensor): the original bounding box in pt form
        label (Tensor): the class label for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, box, class)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, box=None, label=None):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, box, label

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(box, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap < min_iou or max_iou < overlap:
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                              :]

                # keep overlap with gt box IF center in sampled patch
                centers = (box[:2] + box[2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[0]) * (rect[1] < centers[1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[0]) * (rect[3] > centers[1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_box = box.copy()

                # should we use the box left and top corner or the crop's
                current_box[:2] = np.maximum(current_box[:2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_box[:2] -= rect[:2]

                current_box[2:] = np.minimum(current_box[2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_box[2:] -= rect[:2]

                return current_image, current_box, label


class Expand:
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, box, label):
        if random.randint(2):
            return image, box, label

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_image = np.zeros(
            (int(height*ratio), int(width*ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image

        box = box.copy()
        box[:2] += (int(left), int(top))
        box[2:] += (int(left), int(top))

        return image, box, label


class RandomMirror:
    def __call__(self, image, box, label):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            box = box.copy()
            box[0::2] = width - box[2::-2]
        return image, box, label


class SwapChannels:
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort:
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, box, label):
        im = image.copy()
        im, box, label = self.rand_brightness(im, box, label)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, box, label = distort(im, box, label)
        return self.rand_light_noise(im, box, label)


class Augmentation:
    def __init__(self, width, height, mean = (113, 116, 105),
                    std = (55, 51, 58)):
        self.mean = mean
        self.std = std
        self.width = width
        self.height = height
        self.augment = Compose([
            ConvertFromInts(),
            ToAbsoluteCoords(),
            PhotometricDistort(),
            Expand(self.mean),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(self.width, self.height),
            # SubtractMeans(self.mean)
            # Normalize(self.mean, self.std)
            MinMaxNormalize()
        ])

    def __call__(self, img, box, label):
        img, box, label = self.augment(img, box, label)
        # box.clip(max=1.0, min=0.0)
        return img, box, label

class BaseTransform:
    def __init__(self, width, height, mean = (113, 116, 105),
                    std = (55, 51, 58)):
        self.mean = mean
        self.std = std
        self.width = width
        self.height = height
        self.base_transform = Compose([
            Resize(self.width, self.height),
            # SubtractMeans(self.mean)
            # Normalize(self.mean, self.std)
            MinMaxNormalize()
        ])
    
    def __call__(self, img, box, label):
        img, box, label = self.base_transform(img, box, label)
        return img, box, label

if __name__ == '__main__':
    from dataset import DACDataset
    root = "/Share/DAC2020/dataset"
    dataset = DACDataset(root, "train", Augmentation(320, 160))
    img, bbox, label = dataset[0]
    print(img.shape)
    print(type(img))
    print(bbox)
    print(label)