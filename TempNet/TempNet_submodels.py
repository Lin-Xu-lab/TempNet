
"""
Created on June 12th, 2025
@author: Lexie Hassien
"""


import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn.parameter import Parameter
# from torch.nn.modules.module import Module
from torch.utils.data import Dataset

from gradient_reversal import revgrad
from TempNet_main_GitHub.TempNet_utils import *

class Discriminator(nn.Module):
    def __init__(self,embed_dim, n_tm):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, n_tm),
        )
    def forward(self, d):
        dm = self.model(d)
        return dm


class Predictor(nn.Module):
    def __init__(self,embed_dim, n_classes, use_rev=False, rev_alpha=1):
        super(Predictor, self).__init__()
        self.use_rev = use_rev
        self.rev_alpha = torch.tensor(rev_alpha)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )
    def forward(self, x):
        if self.use_rev == True:
            out = revgrad(x, self.rev_alpha)
        else:
            out = x
        out = self.predictor(out)
        return out
    

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
 
    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))
 
    def __len__(self):
        return len(self.data)
 
 
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, groups,  downsample,  use_bn, use_do, dropout = 0.2, is_first_block=False):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do
        self.dropout = dropout
 
        # the first conv
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.do1 = nn.Dropout(p=self.dropout)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            dilation = self.dilation,
            groups=self.groups,
            padding = 'same')
 
        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.do2 = nn.Dropout(p=self.dropout)
        self.conv2 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            dilation = self.dilation,
            groups=self.groups,
            padding = 'same')
 
    def forward(self, x):
        identity = x

        # reference for justifying the order of the layers: https://arxiv.org/pdf/1603.05027
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out) 
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # shortcut
        out += identity
 
        return out 
    
class ResNet1DEncoder2(nn.Module):
    """
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
    Output:
        out: (n_samples)
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNet
        n_block: number of blocks
        n_classes: number of classes
    """
 
    def __init__(self, in_dim, embed_dim, base_filters, kernel_size, stride, groups,
                 n_block, n_classes, dilation = 1, dropout=0.2, downsample_gap=2, use_bn=True, 
                 use_do=True,  verbose=False, **kwargs):
        super(ResNet1DEncoder2, self).__init__(**kwargs)
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do
        self.dropout = dropout
        self.downsample_gap = downsample_gap # 2 for base model
        
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            
            in_channels = 1
            out_channels = 1
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                dilation = self.dilation,
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                dropout = self.dropout,
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)
 
        # final layers
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.LeakyReLU(0.2, inplace=True)
        self.final_pool = nn.AvgPool1d(1,1)
        self.final_dense = nn.Linear(in_dim,embed_dim)
        
    def forward(self, x):

        # input
        out = x
        if self.verbose:
            print('input shape', out.shape)

        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(f'shape after residual block {i_block}: {out.shape}')
 
        ## output embedding
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = self.final_pool(out)
        out = torch.squeeze(out) # remove dimension of size 1
        embed = self.final_dense(out)
 
        return embed
