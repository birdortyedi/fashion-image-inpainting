import torch
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import functional as F
from torchvision.transforms import ToTensor, ToPILImage

from layers import PartialConv2d

import math
import numbers
import warnings
import random


class CentralErasing(object):
    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=None, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")

        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(img_h // 5, img_h - h)  # 0, img_h - h
                j = random.randint(img_w // 4, 3 * img_w // 4 - w)  # 0, img_w - w
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img):
        x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=int(self.value))
        mask = torch.ones_like(img).float()
        mask[:, x:x+h, y:y+w] = 0.0
        return F.erase(img, x, y, h, w, v, self.inplace), \
            ToTensor()(ToPILImage()(mask)),\
            ToTensor()(F.crop(ToPILImage()(img), x, y, h, w))


def unnormalize_batch(batch, mean_, std_, div_factor=1.0):
    """
    Unnormalize batch
    :param batch: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :param div_factor: normalizing factor before data whitening
    :return: unnormalized data, tensor with shape
     (batch_size, nbr_channels, height, width)
    """
    # normalize using dataset mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = mean_[0]
    mean[:, 1, :, :] = mean_[1]
    mean[:, 2, :, :] = mean_[2]
    std[:, 0, :, :] = std_[0]
    std[:, 1, :, :] = std_[1]
    std[:, 2, :, :] = std_[2]
    batch = torch.div(batch, div_factor)

    batch *= Variable(std)
    batch = torch.add(batch, Variable(mean))
    return batch


def weights_init(m):
    if type(m) == PartialConv2d:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    from torchvision.datasets import ImageFolder
    from torch.utils import data
    from torchvision import transforms
    from tqdm import tqdm
    from colorama import Fore

    mask_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomVerticalFlip(p=0.5),
                                         transforms.Resize(256),
                                         transforms.ToTensor(), ])

    m_val = ImageFolder(root="./data/nvidia/", transform=mask_transform)
    val_mask_loader = data.DataLoader(m_val, batch_size=128, shuffle=False, num_workers=0)

    mask_ratios = {"0-10": 0,
                   "10-20": 0,
                   "20-30": 0,
                   "30-40": 0,
                   "40-50": 0,
                   "50-60": 0,
                   "60-70": 0,
                   "70-80": 0,
                   "80-90": 0,
                   "90-100": 0
                   }

    for batch_idx, (mask, _) in tqdm(enumerate(val_mask_loader), ncols=50, desc="Training",
                                     bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        for mask_idx, m in enumerate(mask):
            m = torch.flatten(m[0, :, :])
            print("Mask size: {}".format(m.size()))

            ones = m.sum()
            print("Ones: {}".format(ones))

            ratio = (m.size(0) - ones.item()) / m.size(0)
            print("Ratio: {}".format(ratio))

            if ratio <= 0.1:
                mask_ratios["0-10"] += 1
            elif 0.1 < ratio <= 0.2:
                mask_ratios["10-20"] += 1
            elif 0.2 < ratio <= 0.3:
                mask_ratios["20-30"] += 1
            elif 0.3 < ratio <= 0.4:
                mask_ratios["30-40"] += 1
            elif 0.4 < ratio <= 0.5:
                mask_ratios["40-50"] += 1
            elif 0.5 < ratio <= 0.6:
                mask_ratios["50-60"] += 1
            elif 0.6 < ratio <= 0.7:
                mask_ratios["60-70"] += 1
            elif 0.7 < ratio <= 0.8:
                mask_ratios["70-80"] += 1
            elif 0.8 < ratio <= 0.9:
                mask_ratios["80-90"] += 1
            else:
                mask_ratios["90-100"] += 1

    print(mask_ratios)

