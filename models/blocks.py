from torch import nn
import torch
from torch.nn.functional import interpolate
from utils.image_utils import *


class ContractingBlock(nn.Module):

    def __init__(self,
                 input_ch,
                 out_ch,
                 use_leaky_relu=0.2,
                 use_dropout=0.0,
                 use_bn=True,
                 use_maxp=False,
                 kernel_size=4,
                 stride=2,
                 padding=1):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(out_ch)
        self.use_bn = use_bn
        if use_leaky_relu > 0.0:
            self.activation = nn.LeakyReLU(use_leaky_relu)
        else:
            self.activation = nn.ReLU()
        if use_dropout > 0.0:
            self.dropout = nn.Dropout(use_dropout)
        self.use_dropout = use_dropout
        if use_maxp:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_maxp = use_maxp

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        x = self.activation(x)
        if self.use_dropout > 0.0:
            x = self.dropout(x)
        if self.use_maxp:
            x = self.maxpool(x)
        return x


class ExpandingBlock(nn.Module):

    def __init__(self,
                 input_ch,
                 out_ch,
                 use_leaky_relu=0.2,
                 use_dropout=0.0,
                 use_bn=True,
                 kernel_size=4,
                 stride=2,
                 padding=1):
        super(ExpandingBlock, self).__init__()
        self.t_conv1 = nn.ConvTranspose2d(input_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(out_ch)
        if use_leaky_relu > 0.0:
            self.activation = nn.LeakyReLU(use_leaky_relu)
        else:
            self.activation = nn.ReLU()
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout > 0.0:
            self.dropout = nn.Dropout(use_dropout)
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):

        x = self.t_conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        x = self.activation(x)
        if self.use_dropout > 0.0:
            x = self.dropout(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], 1)
        return x
