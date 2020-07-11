import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from collections import namedtuple
from layers import PartialConv2d, SelfAttention, GaussianNoise


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(num_features=32)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(num_features=64)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(num_features=128)
        self.conv_4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False)
        self.pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = F.leaky_relu(self.conv_0(x), negative_slope=0.2)
        x = F.leaky_relu(self.bn_1(self.conv_1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn_2(self.conv_2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn_3(self.conv_3(x)), negative_slope=0.2)
        x = torch.sigmoid(self.pooling(self.conv_4(x).squeeze()))
        return x


class PConvNet(nn.Module):
    def __init__(self, i_norm=True):
        super(PConvNet, self).__init__()
        self.normalization_layer = nn.InstanceNorm2d if i_norm else nn.BatchNorm2d

        self.block_0 = PartialConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, multi_channel=True, return_mask=True)
        self.block_1 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
        self.norm_1 = self.normalization_layer(num_features=128)
        self.block_2 = PartialConv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2, multi_channel=True, return_mask=True)
        self.norm_2 = self.normalization_layer(num_features=256)
        self.block_3 = PartialConv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
        self.norm_3 = self.normalization_layer(num_features=256)
        self.block_4 = PartialConv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, multi_channel=True, return_mask=True)
        self.norm_4 = self.normalization_layer(num_features=128)

        self.block_6 = PartialConv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1, multi_channel=True, return_mask=True)
        self.norm_6 = self.normalization_layer(num_features=128)
        self.block_7 = PartialConv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1, multi_channel=True, return_mask=True)
        self.norm_7 = self.normalization_layer(num_features=128)
        self.block_8 = PartialConv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1, multi_channel=True, return_mask=True)
        self.norm_8 = self.normalization_layer(num_features=64)
        self.block_9 = PartialConv2d(in_channels=67, out_channels=3, kernel_size=3, padding=1, multi_channel=True, return_mask=True)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)

    def forward(self, x, mask=None):
        x_0, m_0 = self.block_0(x, mask)
        x_0 = F.relu(x_0)
        x_1, m_1 = self.block_1(x_0, m_0)
        x_1 = F.relu(self.norm_1(x_1))
        x_2, m_2 = self.block_2(x_1, m_1)
        x_2 = F.relu(self.norm_2(x_2))
        x_3, m_3 = self.block_3(x_2, m_2)
        x_3 = F.relu(self.norm_3(x_3))
        x_4, m_4 = self.block_4(x_3, m_3)
        x_4 = F.relu(self.norm_4(x_4))

        out = self.upsample(x_4)
        out_mask = self.upsample(m_4)
        out = torch.cat((x_3, out), dim=1)
        out_mask = torch.cat((m_3, out_mask), dim=1)
        out, out_mask = self.block_6(out, out_mask)
        out = F.leaky_relu(self.norm_6(out), negative_slope=0.2)

        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((x_2, out), dim=1)
        out_mask = torch.cat((m_2, out_mask), dim=1)
        out, out_mask = self.block_7(out, out_mask)
        out = F.leaky_relu(self.norm_7(out), negative_slope=0.2)

        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((x_0, out), dim=1)
        out_mask = torch.cat((m_0, out_mask), dim=1)
        out, out_mask = self.block_8(out, out_mask)
        out = F.leaky_relu(self.norm_8(out), negative_slope=0.2)

        out = self.upsample(out)
        out_mask = self.upsample(out_mask)
        out = torch.cat((x, out), dim=1)
        out_mask = torch.cat((mask, out_mask), dim=1)
        out, m_9 = self.block_9(out, out_mask)
        out = torch.tanh(out)

        return out


class Net(nn.Module):
    def __init__(self, i_norm=True, noise=False):
        super(Net, self).__init__()
        self.noise = noise
        self.normalization_layer = nn.InstanceNorm2d if i_norm else nn.BatchNorm2d

        self.block_0 = PartialConv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, multi_channel=True, return_mask=True)
        self.block_1 = PartialConv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
        self.norm_1 = self.normalization_layer(num_features=128)
        self.block_2 = PartialConv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2, multi_channel=True, return_mask=True)
        self.norm_2 = self.normalization_layer(num_features=256)
        self.block_3 = PartialConv2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, multi_channel=True, return_mask=True)
        self.norm_3 = self.normalization_layer(num_features=128)

        self.dilated_block_1 = PartialConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2, multi_channel=True, return_mask=True)
        self.dilated_block_2 = PartialConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=4, dilation=4, multi_channel=True, return_mask=True)
        self.dilated_block_3 = PartialConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=8, dilation=8, multi_channel=True, return_mask=True)
        self.dilated_block_4 = PartialConv2d(in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2, stride=1, multi_channel=True)  # decrease spatial dimension??
        self.dilated_norm_1 = self.normalization_layer(num_features=128)
        self.dilated_norm_2 = self.normalization_layer(num_features=128)
        self.dilated_norm_3 = self.normalization_layer(num_features=128)
        self.dilated_norm_4 = self.normalization_layer(num_features=128)

        # self.s_attention_5 = SelfAttention(in_channels=128)
        # self.block_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.norm_5 = self.normalization_layer(num_features=128)
        self.s_attention_6 = SelfAttention(in_channels=128)
        self.block_6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.norm_6 = self.normalization_layer(num_features=128)
        self.s_attention_7 = SelfAttention(in_channels=384)
        self.block_7 = nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1)
        self.norm_7 = self.normalization_layer(num_features=128)
        self.block_8 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1)
        self.norm_8 = self.normalization_layer(num_features=64)
        self.block_9 = nn.Conv2d(in_channels=67, out_channels=3, kernel_size=3, padding=1)

        self.noise_layer = GaussianNoise()  # noise??
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)

    def forward(self, x, mask=None):
        x_0, m_0 = self.block_0(x, mask)
        x_0 = torch.relu(x_0)
        x_1, m_1 = self.block_1(x_0, m_0)
        x_1 = torch.relu(self.norm_1(x_1))
        x_2, m_2 = self.block_2(x_1, m_1)
        x_2 = torch.relu(self.norm_2(x_2))
        x_3, m_3 = self.block_3(x_2, m_2)
        x_3 = torch.relu(self.norm_3(x_3))

        dilated_x_1, dilated_m_1 = self.dilated_block_1(x_3, m_3)
        dilated_x_1 = torch.relu(self.dilated_norm_1(dilated_x_1))
        dilated_x_2, dilated_m_2 = self.dilated_block_2(dilated_x_1, dilated_m_1)
        dilated_x_2 = torch.relu(self.dilated_norm_2(dilated_m_2))
        dilated_x_2 += dilated_x_1
        dilated_x_3, dilated_m_3 = self.dilated_block_3(dilated_x_2, dilated_m_2)
        dilated_x_3 = torch.relu(self.dilated_norm_3(dilated_x_3))
        dilated_x_3 += dilated_x_2
        dilated_x = torch.relu(self.dilated_norm_4(self.dilated_block_4(dilated_x_3, dilated_m_3)))
        # dilated_x += dilated_x_3

        # x_5_a, _ = self.s_attention_5(dilated_x)
        # x_5_a = torch.relu(x_5_a)

        x_6_a, _ = self.s_attention_6(dilated_x)  # self.s_attention_6(x_5_a)
        x_6_a = torch.relu(x_6_a)
        # x_6 = self.upsample(x_6_a)
        x_3 = torch.cat((x_3, x_6_a), dim=1)  # x_6

        x_6 = self.upsample(x_3)
        x_6 = F.leaky_relu(self.norm_6(self.block_6(x_6)), negative_slope=0.2)
        out = torch.cat((x_2, x_6), dim=1)

        x_7_a, _ = self.s_attention_7(out)
        x_7_a = torch.relu(x_7_a)
        x_7 = self.upsample(x_7_a)
        x_7 = F.leaky_relu(self.norm_7(self.block_7(x_7)), negative_slope=0.2)
        out = torch.cat((x_0, x_7), dim=1)
        out = F.leaky_relu(self.norm_8(self.block_8(out)), negative_slope=0.2)

        x_9 = self.upsample(out)
        out = torch.cat((x, x_9), dim=1)
        out = self.block_9(out)
        out = torch.tanh(out)

        return out


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
