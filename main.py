import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from PIL import Image
from tqdm import tqdm
from colorama import Fore

import os
import numpy as np

from utils import weights_init, unnormalize_batch
from datasets import FashionGen, FashionAI, DeepFashion, DeepFashion2, CelebAHQ
from models import Net, PConvNet, Discriminator, VGG16
from losses import CustomLoss

NUM_EPOCHS = 21
BATCH_SIZE = 16
DATASET = "DeepFashion2"
DATA_FOLDER = os.path.join("./data", DATASET)
MODE = "ablation"
MASK_FORM = "free"  # "free"
MULTI_GPU = True
DEVICE_ID = 1
DEVICE = "cuda" if MULTI_GPU else "cuda:{}".format(DEVICE_ID)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

if DATASET == "FashionGen":
    train_dataset = FashionGen(filename=os.path.join(DATA_FOLDER, "fashiongen_256_256_train.h5"), mask_form=MASK_FORM)
    val_dataset = FashionGen(filename=os.path.join(DATA_FOLDER, "fashiongen_256_256_validation.h5"), mask_form=MASK_FORM)
elif DATASET == "FashionAI":
    train_dataset = FashionAI(filename=os.path.join(DATA_FOLDER, "train", "Numpys", "train_imgs.npy"), mask_form=MASK_FORM)
    val_dataset = FashionAI(filename=os.path.join(DATA_FOLDER, "train", "Numpys", "train_imgs.npy"), mask_form=MASK_FORM)
elif DATASET == "DeepFashion":
    train_dataset = DeepFashion(base_folder=DATA_FOLDER, mask_form=MASK_FORM)
    val_dataset = DeepFashion(base_folder=DATA_FOLDER, mode="val", mask_form=MASK_FORM)
elif DATASET == "DeepFashion2":
    df2_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.Resize((256, 256)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(MEAN, STD)])
    train_dataset = ImageFolder(root=os.path.join(DATA_FOLDER, "train"), transform=df2_transform)
    val_dataset = ImageFolder(root=os.path.join(DATA_FOLDER, "validation"), transform=df2_transform)
elif DATASET == "CelebAHQ":
    celebA_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.Resize(256),
                                           transforms.ToTensor(),
                                           transforms.Normalize(MEAN, STD)])
    train_dataset = ImageFolder(root=os.path.join(DATA_FOLDER, MODE, "CelebA-HQ-img/"), transform=celebA_transform)
    val_dataset = ImageFolder(root=os.path.join(DATA_FOLDER, MODE, "CelebA-HQ-img/"), transform=celebA_transform)
else:
    raise NotImplementedError("Unknown dataset.")

print("Sample size in training: {}".format(len(train_dataset)))

train_img_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_img_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

if MASK_FORM == "free":
    mask_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomVerticalFlip(p=0.5),
                                         transforms.Resize(256),
                                         transforms.ToTensor(), ])

    m_train = ImageFolder(root="./data/qd_imd_big/train/", transform=mask_transform)
    m_val = ImageFolder(root="./data/qd_imd_big/test/", transform=mask_transform)

    print("Mask size in training: {}".format(len(m_train)))

    train_mask_loader = data.DataLoader(m_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_mask_loader = data.DataLoader(m_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

d_net = Discriminator()
refine_net = Net()

if MODE != "ablation":
    vgg = VGG16(requires_grad=False)
    vgg.to(device)

if torch.cuda.device_count() > 1 and MULTI_GPU:
    print("Using {} GPUs...".format(torch.cuda.device_count()))
    d_net = nn.DataParallel(d_net)
    refine_net = nn.DataParallel(refine_net)
else:
    print("GPU ID: {}".format(device))

d_net = d_net.to(device)
refine_net = refine_net.to(device)

if MODE == "train":
    refine_net.apply(weights_init)

d_loss_fn = nn.BCELoss()
d_loss_fn = d_loss_fn.to(device)
refine_loss_fn = CustomLoss()
refine_loss_fn = refine_loss_fn.to(device)

lr, r_lr, d_lr = 0.0004, 0.0004, 0.0004
d_optimizer = optim.Adam(d_net.parameters(), lr=d_lr, betas=(0.9, 0.999))
r_optimizer = optim.Adam(refine_net.parameters(), lr=r_lr, betas=(0.5, 0.999))

d_scheduler = optim.lr_scheduler.ExponentialLR(d_optimizer, gamma=0.9)
r_scheduler = optim.lr_scheduler.ExponentialLR(r_optimizer, gamma=0.9)

if MODE != "ablation":
    writer = SummaryWriter()


def train(epoch, img_loader, mask_loader=None):
    for batch_idx, (y_train, _) in tqdm(enumerate(img_loader), ncols=50, desc="Training", total=len(train_dataset) // BATCH_SIZE,
                                                  bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.GREEN, Fore.RESET)):
        if MASK_FORM == "free":
            x_mask, _ = next(iter(mask_loader))
            if x_mask.size(0) != y_train.size(0):
                x_mask = x_mask[:y_train.size(0)]

            x_train = x_mask * y_train + (1.0 - x_mask) * 0.5

        num_step = epoch * len(img_loader) + batch_idx

        x_mask = x_mask.float().to(device)
        y_train = unnormalize_batch(y_train, MEAN, STD).float().to(device)
        x_train = x_train.float().to(device)

        writer.add_scalar("LR/learning_rate", r_scheduler.get_lr(), num_step)

        refine_net.zero_grad()
        r_output = refine_net(x_train, x_mask)
        d_output = d_net(r_output.detach()).view(-1)
        r_composite = x_mask * y_train + (1.0 - x_mask) * r_output

        vgg_features_gt = vgg(y_train)
        vgg_features_composite = vgg(r_composite)
        vgg_features_output = vgg(r_output)

        r_total_loss, r_pixel_loss, r_perceptual_loss, \
            adversarial_loss, r_tv_loss = refine_loss_fn(y_train, r_output, r_composite, d_output,
                                                         vgg_features_gt, vgg_features_output, vgg_features_composite)

        writer.add_scalar("Refine_G/on_step_total_loss", r_total_loss.item(), num_step)
        writer.add_scalar("Refine_G/on_step_pixel_loss", r_pixel_loss.item(), num_step)
        writer.add_scalar("Refine_G/on_step_perceptual_loss", r_perceptual_loss.item(), num_step)
        writer.add_scalar("Refine_G/on_step_adversarial_loss", adversarial_loss.item(), num_step)
        writer.add_scalar("Refine_G/on_step_tv_loss", r_tv_loss.item(), num_step)

        r_total_loss.backward()
        r_optimizer.step()

        d_net.zero_grad()
        d_real_output = d_net(y_train).view(-1)
        d_fake_output = d_output.detach()

        if torch.rand(1) > 0.1:
            d_real_loss = d_loss_fn(d_real_output, torch.FloatTensor(d_real_output.size(0)).uniform_(0.0, 0.3).to(device))
            d_fake_loss = d_loss_fn(d_fake_output, torch.FloatTensor(d_fake_output.size(0)).uniform_(0.7, 1.2).to(device))
        else:
            d_real_loss = d_loss_fn(d_real_output, torch.FloatTensor(d_fake_output.size(0)).uniform_(0.7, 1.2).to(device))
            d_fake_loss = d_loss_fn(d_fake_output, torch.FloatTensor(d_real_output.size(0)).uniform_(0.0, 0.3).to(device))

        writer.add_scalar("Discriminator/on_step_real_loss", d_real_loss.mean().item(), num_step)
        writer.add_scalar("Discriminator/on_step_fake_loss", d_fake_loss.mean().item(), num_step)

        d_loss = d_real_loss + d_fake_loss

        d_loss.backward()
        d_optimizer.step()

        if batch_idx % 1000 == 0:
            x_grid = make_grid(unnormalize_batch(x_train[:6], MEAN, STD), nrow=6, padding=2)
            y_grid = make_grid(y_train[:6], nrow=6, padding=2)
            r_output_grid = make_grid(r_output[:6], nrow=6, padding=2)
            r_composite_grid = make_grid(r_composite[:6], nrow=6, padding=2)

            writer.add_image("x_train/epoch_{}".format(epoch), x_grid, num_step)
            writer.add_image("org/epoch_{}".format(epoch), y_grid, num_step)
            writer.add_image("refine_output/epoch_{}".format(epoch), r_output_grid, num_step)
            writer.add_image("refine_composite/epoch_{}".format(epoch), r_composite_grid, num_step)

            print("Step:{}  ".format(num_step),
                  "Epoch:{}".format(epoch),
                  "[{}/{} ".format(batch_idx * len(x_train), len(train_img_loader.dataset)),
                  "({}%)]  ".format(int(100 * batch_idx / float(len(train_img_loader))))
                  )


def ablation(path):
    normalizer = transforms.Normalize(MEAN, STD)
    tensorizer = transforms.ToTensor()
    pillowize = transforms.ToPILImage()
    resizer = transforms.Resize((256, 256))
    refine_net.eval()
    with torch.no_grad():
        imgs, masks, idxs = list(), list(), list()
        for i, fname in enumerate(os.listdir(os.path.join(path, "img"))):
            if os.path.isfile(os.path.join(path, "masked", "mask_" + fname)):
                idxs.append(i)
                imgs.append(normalizer(tensorizer(resizer(Image.open(os.path.join(path, "img", fname))))))
                m = tensorizer(resizer(Image.open(os.path.join(path, "masked", "mask_" + fname))))
                masks.append(m)

        new_masks = list()
        for ma in masks:
            if ma.size(0) == 1:
                ma = torch.cat([ma] * 3)
            new_masks.append(ma)
        masks = new_masks

        imgs = torch.stack(imgs).float().to(device)
        masks = torch.stack(masks).float().to(device)

        x_train = masks * imgs + (1.0 - masks) * 0.5
        output = refine_net(x_train, masks)
        output = masks * unnormalize_batch(imgs, MEAN, STD) + (1.0 - masks) * output
        fnames = os.listdir(os.path.join(path, "img"))
        x_train = unnormalize_batch(x_train, MEAN, STD)
        for out, x, idx in zip(output, x_train, idxs):
            pillowize(x.squeeze().cpu()).save(os.path.join(path, "train", "train_" + fnames[idx]))
            pillowize(out.squeeze().cpu()).save(os.path.join(path, "res", "res_" + fnames[idx]))


if __name__ == '__main__':
    if MODE == "train":
        if not os.path.exists("./weights_{}_{}".format(DATASET, MASK_FORM)):
            os.mkdir("./weights_{}_{}".format(DATASET, MASK_FORM))

        for e in range(NUM_EPOCHS):
            if MASK_FORM == "free":
                train(e, train_img_loader, train_mask_loader)
            else:
                train(e, train_img_loader)
            r_scheduler.step(e)
            d_scheduler.step(e)
            torch.save(refine_net.state_dict(), "./weights_{}_{}/weights_net_epoch_{}.pth".format(DATASET, MASK_FORM, e))
        writer.close()
    else:
        PATH = "./weights/weights_{}_{}/weights_net_epoch_{}.pth".format(DATASET, MASK_FORM, NUM_EPOCHS)
        refine_net.load_state_dict(torch.load(PATH))
        ABLATION_DATA_PATH = "./data/ablation"
        ablation(ABLATION_DATA_PATH)
