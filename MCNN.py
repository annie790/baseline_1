import torch
import torch.nn as nn
#from network import Conv2d

import pdb

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MCNN(nn.Module):
    '''
    Multi-column CNN 
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''
    
    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        
        self.branch1 = nn.Sequential(Conv2d( 3, 16, 9, same_padding=True, bn=None),
                                     nn.InstanceNorm2d(16),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=None),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=None),
                                     nn.InstanceNorm2d(16),
                                     Conv2d(16,  8, 7, same_padding=True, bn=None),
                                     nn.InstanceNorm2d(8))
        
        self.branch2 = nn.Sequential(Conv2d( 3, 20, 7, same_padding=True, bn=None),
                                     nn.InstanceNorm2d(20),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=None),
                                     nn.InstanceNorm2d(40),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=None),
                                     nn.InstanceNorm2d(20),
                                     Conv2d(20, 10, 5, same_padding=True, bn=None),
                                     nn.InstanceNorm2d(10))
        
        self.branch3 = nn.Sequential(Conv2d( 3, 24, 5, same_padding=True, bn=None),
                                     nn.InstanceNorm2d(24),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=None),
                                     nn.InstanceNorm2d(48),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=None),
                                     nn.InstanceNorm2d(24),
                                     Conv2d(24, 12, 3, same_padding=True, bn=None),
                                     nn.InstanceNorm2d(12))
        
        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=None), nn.InstanceNorm2d(1))
        self.fuse1 = nn.Sequential(Conv2d( 8, 1, 1, same_padding=True, bn=None), nn.InstanceNorm2d(1))
        self.fuse2 = nn.Sequential(Conv2d( 10, 1, 1, same_padding=True, bn=None), nn.InstanceNorm2d(1))
        self.fuse3 = nn.Sequential(Conv2d( 12, 1, 1, same_padding=True, bn=None), nn.InstanceNorm2d(1))
        self.unpool2 = nn.Upsample(scale_factor=2, mode='bilinear')        
    def forward(self, im_data):
        #pdb.set_trace()
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)
        x = torch.cat((x1,x2,x3),1)
        x = self.fuse(x)
        x = self.unpool2(x)
        x = self.unpool2(x)
        x = x.view(-1,x.size(2),x.size(3))



        # y1 = self.fuse1(x1)
        # y2 = self.fuse2(x2)
        # y3 = self.fuse3(x3)
        # y1 = self.unpool2(y1)
        # y2 = self.unpool2(y2)
        # y3 = self.unpool2(y3)
        # y1 = self.unpool2(y1)
        # y2 = self.unpool2(y2)
        # y3 = self.unpool2(y3)
        # y1 = y1.view(-1,y1.size(2),y1.size(3))
        # y2 = y2.view(-1,y2.size(2),y2.size(3))
        # y3 = y3.view(-1,y3.size(2),y3.size(3))

        #return x,y1,y2,y3
        return x

def conv3x3(in_planes, out_planes, strd=1, padding=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding)

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.conv = conv3x3(in_planes, out_planes)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

