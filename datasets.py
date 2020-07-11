from torchvision import transforms
from torch.utils import data

from utils import CentralErasing

from PIL import Image

import os
import re
import csv
import h5py
import glob
import shutil
import random
import numpy as np


class FashionGen(data.Dataset):
    categories = ['TOPS', 'SWEATERS', 'PANTS', 'JEANS', 'SHIRTS', 'DRESSES', 'SHORTS', 'SKIRTS', 'JACKETS & COATS']

    def __init__(self, filename, mask_form=None, is_train=True):
        super().__init__()
        self.h5_file = h5py.File(filename, mode="r")
        self.is_train = is_train
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        if mask_form is None:
            self.mask_form = "rectangular"
        else:
            self.mask_form = mask_form

        self.indices = list(i for i, c in enumerate(self.h5_file['input_category'][:]) if c[0].decode("latin-1") in self.categories)

        self.h_flipper = transforms.RandomHorizontalFlip(p=0.5)
        self.normalizer = transforms.Normalize(self.mean, self.std)

    def __getitem__(self, index):
        i = self.indices[index]
        img = self.h5_file["input_image"][i, :, :]
        img = transforms.ToPILImage()(img)

        if self.is_train:
            img = self.h_flipper(img)

        img = transforms.ToTensor()(img)

        if self.mask_form == "rectangular":
            rnd_central_eraser = CentralErasing(scale=(0.0625, 0.125), ratio=(0.75, 1.5), value=1)
            erased, mask, _ = rnd_central_eraser(img)
            erased = self.normalizer(erased)
        else:
            erased, mask = img, list()
            erased = self.normalizer(erased)

        return img, erased, mask

    def __len__(self):
        return len(self.indices)


class FashionAI(data.Dataset):
    def __init__(self, filename, csv_file=None, size=256, data_folder=None, mask_form=None, load_from_file=True, mode="train"):
        super(FashionAI, self).__init__()
        self.csv_file = csv_file
        self.data_folder = data_folder
        self.size = size
        self.mode = mode

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.imgs = np.load(filename) if load_from_file else self.load_data()

        if mask_form is None:
            self.mask_form = "rectangular"
        else:
            self.mask_form = mask_form

        self.h_flipper = transforms.RandomHorizontalFlip(p=0.5)
        self.normalizer = transforms.Normalize(self.mean, self.std)

    def __getitem__(self, index):
        img = self.imgs[index]
        img = transforms.ToPILImage()(img)

        if self.mode == "train":
            img = self.h_flipper(img)

        img = transforms.ToTensor()(img)

        if self.mask_form == "rectangular":
            rnd_central_eraser = CentralErasing(scale=(0.0625, 0.125), ratio=(0.75, 1.5), value=1)
            erased, mask, _ = rnd_central_eraser(img)
            erased = self.normalizer(erased)
        else:
            erased, mask = img, list()
            erased = self.normalizer(erased)

        return img, erased, mask

    def __len__(self):
        return len(self.imgs)

    def read_csv(self):
        data = list()
        with open(self.csv_file) as f:
            reader = csv.reader(f)
            for row in reader:
                row_dict = dict()
                row_dict["file"] = row[0]
                data.append(row_dict)
        return data

    def load_data(self):
        data = self.read_csv()
        imgs = list()
        for row in data:
            img_file = os.path.join(self.data_folder, self.mode, row["file"])
            imgs.append(np.array(Image.open(img_file).convert("RGB").resize((self.size, self.size))))
        return np.array(imgs)

    def save(self):
        print("Start reading dataset FashionAI in mode {}".format(self.mode))
        imgs = self.load_data()
        random.shuffle(imgs)
        print("Shape: {}".format(imgs.shape))
        print("Saving in {}...".format(self.mode))
        with h5py.File(os.path.join(self.data_folder, "fashionAI_{}_{}_{}.h5".format(self.size, self.size, self.mode)), "w") as f:
            f.create_dataset("input_image", data=imgs, compression="gzip", compression_opts=9, dtype="uint8")
        print("Done for FashionAI")


class DeepFashion(data.Dataset):
    def __init__(self, size=256, base_folder=None, anno_folder="anno", data_folder=None,
                 mask_form=None, mode="train", processed=True):
        super(DeepFashion, self).__init__()
        self.size = size
        self.base_folder = base_folder
        self.anno_folder = os.path.join(self.base_folder, anno_folder)
        self.data_folder = data_folder if data_folder is not None else os.path.join(self.base_folder, "img")
        self.mode = mode

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        self.splitter = re.compile("\s+")

        self.imgs = self.load_data()

        if mask_form is None:
            self.mask_form = "rectangular"
        else:
            self.mask_form = mask_form

        if not processed:
            self.process_folders()

        self.h_flipper = transforms.RandomHorizontalFlip(p=0.5)
        self.center_cropper = transforms.CenterCrop(size=200)
        self.resizer = transforms.Resize((self.size, self.size))
        self.normalizer = transforms.Normalize(self.mean, self.std)

    def __getitem__(self, index):
        img = self.imgs[index]
        img = transforms.ToPILImage()(img)

        if self.mode == "train":
            img = self.h_flipper(img)

        img = self.center_cropper(img)
        img = self.resizer(img)
        img = transforms.ToTensor()(img)

        if self.mask_form == "rectangular":
            rnd_central_eraser = CentralErasing(scale=(0.0625, 0.125), ratio=(0.75, 1.5), value=1)
            erased, mask, _ = rnd_central_eraser(img)
            erased = self.normalizer(erased)
        else:
            erased, mask = img, list()
            erased = self.normalizer(erased)

        return img, erased, mask

    def __len__(self):
        return len(self.imgs)

    def load_data(self):
        imgs = list()
        for img_file in glob.glob(os.path.join(self.data_folder, self.mode, "**/*.jpg"), recursive=True):
            img = Image.open(img_file).convert("RGB")
            imgs.append(np.array(img))
        return np.array(imgs)

    def save(self):
        print("Start reading dataset DeepFashion in mode {}".format(self.mode))
        imgs = self.load_data()
        random.shuffle(imgs)
        print("Shape: {}".format(imgs.shape))
        print("Saving in {}...".format(self.mode))
        with h5py.File(os.path.join(self.base_folder, "deepfashion_{}_{}_{}.h5".format(self.size, self.size, self.mode)), "w") as f:
            f.create_dataset("input_image", data=imgs, compression="gzip", compression_opts=9, dtype="uint8")
        print("Done for DeepFashion")

    def process_folders(self):
        with open(os.path.join(self.anno_folder, "list_eval_partition.txt"), 'r') as eval_partition_file:
            list_eval_partition = [line.rstrip('\n') for line in eval_partition_file][2:]
            list_eval_partition = [self.splitter.split(line) for line in list_eval_partition]
            list_all = [(v[0][4:], v[1]) for v in list_eval_partition]

        for element in list_all:
            if not os.path.exists(os.path.join(self.data_folder, element[1])):
                os.mkdir(os.path.join(self.data_folder, element[1]))
            if not os.path.exists(os.path.join(self.data_folder, element[1], element[0].split('/')[0])):
                os.mkdir(os.path.join(self.data_folder, element[1], element[0].split('/')[0]))
            if os.path.exists(os.path.join(self.data_folder, element[0])):
                shutil.move(os.path.join(self.data_folder, element[0]), os.path.join(self.data_folder, element[1], element[0].split('/')[0]))


class DeepFashion2(data.Dataset):
    def __init__(self, size=256, data_folder=None, mode="train", mask_form=None):
        super(DeepFashion2, self).__init__()
        self.size = size
        self.data_folder = data_folder if data_folder is None else os.path.join(data_folder, mode)
        self.mode = mode

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        if mask_form is None:
            self.mask_form = "rectangular"
        else:
            self.mask_form = mask_form

        self.h_flipper = transforms.RandomHorizontalFlip(p=0.5)
        if self.mode == "train":
            self.cropper = transforms.RandomCrop(size=224)
        else:
            self.cropper = transforms.CenterCrop(size=224)
        self.resizer = transforms.Resize(size=256)
        self.normalizer = transforms.Normalize(self.mean, self.std)

        self.imgs = self.load_data()

    def __getitem__(self, index):
        img = self.imgs[index]
        img = transforms.ToPILImage()(img)

        if self.mode == "train":
            img = self.h_flipper(img)

        img = self.cropper(img)
        img = self.resizer(img)

        img = transforms.ToTensor()(img)

        if self.mask_form == "rectangular":
            rnd_central_eraser = CentralErasing(scale=(0.0625, 0.125), ratio=(0.75, 1.5), value=1)
            erased, mask, _ = rnd_central_eraser(img)
            erased = self.normalizer(erased)
        else:
            erased, mask = img, list()
            erased = self.normalizer(erased)

        return img, erased, mask

    def __len__(self):
        return len(self.imgs)

    def load_data(self):
        imgs = list()
        for img_file in glob.glob(os.path.join(self.data_folder, "**/*.jpg"), recursive=True):
            img = Image.open(img_file).convert("RGB")

            if self.mode == "train":
                img = self.h_flipper(img)

            # img = self.cropper(img)
            img = self.resizer(img)
            imgs.append(np.array(img))
        return np.array(imgs)

    def save(self):
        print("Start reading dataset DeepFashion2 in mode {}".format(self.mode))
        imgs = self.load_data()
        random.shuffle(imgs)
        print("Shape: {}".format(imgs.shape))
        print("Saving in {}...".format(self.mode))
        with h5py.File(os.path.join(self.data_folder, "deepfashion2_{}_{}_{}.h5".format(self.size, self.size, self.mode)), "w") as f:
            f.create_dataset("input_image", data=imgs, compression="gzip", compression_opts=9, dtype="uint8")
        print("Done for DeepFashion2")


class CelebAHQ(data.Dataset):
    def __init__(self, size=256, data_folder=None, mode="train", mask_form=None):
        super(CelebAHQ, self).__init__()
        self.size = size
        self.data_folder = data_folder if data_folder is None else os.path.join(data_folder, mode)
        self.mode = mode

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

        if mask_form is None:
            self.mask_form = "rectangular"
        else:
            self.mask_form = mask_form

        self.h_flipper = transforms.RandomHorizontalFlip(p=0.5)
        self.normalizer = transforms.Normalize(self.mean, self.std)

        self.imgs = self.load_data()

    def __getitem__(self, index):
        img = self.imgs[index]
        img = transforms.ToPILImage()(img)

        if self.mode == "train":
            img = self.h_flipper(img)

        img = transforms.ToTensor()(img)

        if self.mask_form == "rectangular":
            rnd_central_eraser = CentralErasing(scale=(0.0625, 0.125), ratio=(0.75, 1.5), value=1)
            erased, mask, _ = rnd_central_eraser(img)
            erased = self.normalizer(erased)
        else:
            erased, mask = img, list()
            erased = self.normalizer(erased)

        return img, erased, mask

    def __len__(self):
        return len(self.imgs)

    def load_data(self):
        imgs = list()
        for img_file in glob.glob(os.path.join(self.data_folder, "**/*.jpg"), recursive=True):
            img = Image.open(img_file).convert("RGB")

            if self.mode == "train":
                img = self.h_flipper(img)

            imgs.append(np.array(img))
        return np.array(imgs)


if __name__ == '__main__':
    # fashionAI_train = FashionAI(filename=None,
    #                             csv_file=os.path.join("./data", "FashionAI", "train", "Annotations/results.csv"),
    #                             data_folder=os.path.join("./data", "FashionAI"), mask_form="free")
    # fashionAI_train.save()
    # del fashionAI_train
    # fashionAI_val = FashionAI(filename=None,
    #                           csv_file=os.path.join("./data", "FashionAI", "train", "Annotations/results.csv"),
    #                           data_folder=os.path.join("./data", "FashionAI"), mask_form="free", mode="train")

    deepfashion_train = DeepFashion(base_folder=os.path.join("./data", "DeepFashion"), mask_form="free")
    deepfashion_train.save()
    del deepfashion_train
    deepfashion_validation = DeepFashion(base_folder=os.path.join("./data", "DeepFashion"), mask_form="free", mode="validation")
    deepfashion_validation.save()
    del deepfashion_validation

    deepfashion2_train = DeepFashion2(data_folder=os.path.join("./data", "DeepFashion2"), mask_form="free")
    deepfashion2_train.save()
    del deepfashion2_train
    deepfashion2_validation = DeepFashion2(data_folder=os.path.join("./data", "DeepFashion2"), mask_form="free", mode="validation")
    deepfashion2_validation.save()
    del deepfashion2_validation
