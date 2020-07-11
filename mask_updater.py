from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
import torch.nn.functional as F
import torch

from tqdm import tqdm
from colorama import Fore
import numpy as np

from layers import PartialConv2d

pconv = PartialConv2d(in_channels=3, out_channels=3, kernel_size=9, stride=1, padding=0, multi_channel=True, return_mask=True)
pdconv = PartialConv2d(in_channels=3, out_channels=3, kernel_size=5, padding=4, dilation=4, multi_channel=True, return_mask=True)

mask_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5),
                                     transforms.Resize(256),
                                     transforms.ToTensor(), ])

m_train = ImageFolder(root="./data/nvidia/", transform=mask_transform)

print("Mask size in training: {}".format(len(m_train)))

train_mask_loader = data.DataLoader(m_train, batch_size=1, shuffle=True, num_workers=0)


def observe(mask_ratio):
    lst, lst_d = list(), list()
    for batch_idx, (x_mask, _) in tqdm(enumerate(train_mask_loader), ncols=50, desc="Training",
                                       bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        x_mask = 1 - x_mask
        masks = x_mask.size()
        ratio = x_mask.sum() / (masks[1] * masks[2] * masks[3])
        if not (mask_ratio[0] < ratio <= mask_ratio[1]):
            continue
        x_mask_d = x_mask.clone()
        is_ok, is_ok_d = False, False
        for i in range(20):
            masks = x_mask.size()
            masks_d = x_mask_d.size()
            ratio = x_mask.sum() / (masks[1] * masks[2] * masks[3])
            ratio_d = x_mask_d.sum() / (masks_d[1] * masks_d[2] * masks_d[3])
            _, x_mask = pconv(x_mask, x_mask)
            _, x_mask_d = pdconv(x_mask_d, x_mask_d)
            if ratio == 1.0:
                if not is_ok:
                    lst.append(i)
                    is_ok = True
            if ratio_d == 1.0:
                if not is_ok_d:
                    lst_d.append(i)
                    is_ok_d = True

        # if batch_idx % 100 == 0:
        #    print("")
        #    print(sum(lst) / len(lst))
        #    print(sum(lst_d) / len(lst_d))

    return lst, lst_d


if __name__ == '__main__':
    ratios = dict()
    for i in [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]:
        m_lst, md_lst = observe(i)
        avg_m_lst = sum(m_lst) / len(m_lst)
        avg_md_lst = sum(md_lst) / len(md_lst)
        ratios[i] = {"range": i, "m_size": len(m_lst), "md_size": len(md_lst),
                     "avg_m": avg_m_lst, "avg_md": avg_md_lst}
        print(ratios[i])
    print("Done")
    print(ratios)



