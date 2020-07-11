import torch
import torch.nn.functional as F
from torch import nn


class PartialConv2d(nn.Conv2d):
    ###############################################################################
    # BSD 3-Clause License
    #
    # Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
    #
    # Author & Contact: Guilin Liu (guilinl@nvidia.com)
    ###############################################################################

    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, x, mask_in=None):
        assert len(x.shape) == 4
        if mask_in is not None or self.last_size != tuple(x.shape):
            self.last_size = tuple(x.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(x.data.shape[0], x.data.shape[1], x.data.shape[2], x.data.shape[3]).to(x)
                    else:
                        mask = torch.ones(1, 1, x.data.shape[2], x.data.shape[3]).to(x)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-6)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = super(PartialConv2d, self).forward(torch.mul(x, mask) if mask_in is not None else x)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class SelfAttention(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_channels, gamma=1):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels

        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(gamma), requires_grad=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N: width*height)
        """
        b_size, C, w, h = x.size()
        p_query = self.query_conv(x).view(b_size, -1, w * h).permute(0, 2, 1)  # B X C X (N)
        p_key = self.key_conv(x).view(b_size, -1, w * h)  # B X C x (*W*H)
        energy = torch.bmm(p_query, p_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        p_value = self.value_conv(x).view(b_size, -1, w * h)  # B X C X N

        out = torch.bmm(p_value, attention.permute(0, 2, 1))
        out = out.view(b_size, C, w, h)

        out = self.gamma * out + x
        return out, attention


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x, noise):
        if self.training and self.sigma != 0.0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = noise * scale
            x = x + sampled_noise
        return x
