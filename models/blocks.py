from torch import nn
import torch
from torch.nn.functional import interpolate
from utils.image_utils import *

class ContractingBlock(nn.Module):
    '''
    ContractingBlock Class
    Performs two convolutions followed by a max pool operation.
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_ch, out_ch, use_leakyRelu = 0.2, use_dropout=0.0, use_bn=True,use_maxp = False, kernel_size = 4 ,stride = 2, padding = 1):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, out_ch, kernel_size=kernel_size,stride = stride, padding=padding)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(out_ch)
        self.use_bn = use_bn
        if use_leakyRelu > 0.0:
            self.activation = nn.LeakyReLU(use_leakyRelu)
        else:
            self.activation = nn.ReLU()
        if use_dropout > 0.0:
            self.dropout = nn.Dropout(use_dropout)
        self.use_dropout = use_dropout
        if use_maxp:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.use_maxp = use_maxp
    def forward(self, x):
        '''
        Function for completing a forward pass of ContractingBlock:
        Given an image tensor, completes a contracting block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
        '''
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
    '''
    ExpandingBlock Class:
    Performs an upsampling, a convolution, a concatenation of its two inputs,
    followed by two more convolutions with optional dropout
    Values:
        input_channels: the number of channels to expect from a given input
    '''
    def __init__(self, input_ch, out_ch, use_leakyRelu = 0.2, use_dropout=0.0, use_bn=True, kernel_size = 4 ,stride = 2, padding = 1):
        super(ExpandingBlock, self).__init__()
        self.t_conv1 = nn.ConvTranspose2d(input_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(out_ch)
        if use_leakyRelu > 0.0:
            self.activation = nn.LeakyReLU(use_leakyRelu)
        else:
            self.activation = nn.ReLU()
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout > 0.0:
            self.dropout = nn.Dropout(use_dropout)
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        '''
        Function for completing a forward pass of ExpandingBlock:
        Given an image tensor, completes an expanding block and returns the transformed tensor.
        Parameters:
            x: image tensor of shape (batch size, channels, height, width)
            skip_con_x: the image tensor from the contracting path (from the opposing block of x)
                    for the skip connection
        '''
        x = self.t_conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        x = self.activation(x)
        if self.use_dropout > 0.0:
            x = self.dropout(x)
        skip_con_x = crop(skip_con_x, x.shape)
        x = torch.cat([x, skip_con_x], axis=1)
        return x
