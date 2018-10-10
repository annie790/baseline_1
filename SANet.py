import torch
import torch.nn as nn


import pdb

class Conv2d(nn.Module):
    def __init__(self, kernel_size,in_channels, out_channels, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.bn = nn.InstanceNorm2d(out_channels, eps=0.00001, momentum=0.1, affine=True) if bn else None

        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Trans_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,output_padding,relu=True, bn=False):
        super(Trans_Conv2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups=1, bias=True, dilation=1)
        # self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.bn = nn.InstanceNorm2d(out_channels, eps=0.00001, momentum=0.1, affine=True) if bn else None

        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SANet(nn.Module):
    '''
    Multi-column CNN
        -Implementation of Single Image Crowd Counting via Multi-column CNN (Zhang et al.)
    '''

    def __init__(self, bn=False):
        super(SANet, self).__init__()

        ####first scale no conv1*1
        self.branch1_1 = nn.Sequential(Conv2d( 1, 3, 16, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))

        self.branch1_2 = nn.Sequential(Conv2d( 3, 3, 16, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))

        self.branch1_3 = nn.Sequential(Conv2d( 5, 3, 16, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))

        self.branch1_4 = nn.Sequential(Conv2d( 7, 3, 16, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))

        ##second and after conv1*1 c---c/2
        self.branch2_1 = nn.Sequential(Conv2d( 1, 64, 32, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))

        self.branch2_2 = nn.Sequential(Conv2d( 1, 64, 32, same_padding=True, bn=True),
                                     Conv2d( 3, 32, 32, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))

        self.branch2_3 = nn.Sequential(Conv2d( 1, 64, 32, same_padding=True, bn=True),
                                     Conv2d( 5, 32, 32, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))

        self.branch2_4 = nn.Sequential(Conv2d( 1, 64, 32, same_padding=True, bn=True),
                                     Conv2d( 7, 32, 32, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))
        ##second and after conv1*1 c---c/2
        self.branch3_1 = nn.Sequential(Conv2d( 1, 128, 32, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))

        self.branch3_2 = nn.Sequential(Conv2d( 1, 128, 64, same_padding=True, bn=True),
                                     Conv2d( 3, 64, 32, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))

        self.branch3_3 = nn.Sequential(Conv2d( 1, 128, 64, same_padding=True, bn=True),
                                     Conv2d( 5, 64, 32, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))

        self.branch3_4 = nn.Sequential(Conv2d( 1, 128, 64, same_padding=True, bn=True),
                                     Conv2d( 7, 64, 32, same_padding=True, bn=True),
                                     nn.MaxPool2d(2))

        ##second and after conv1*1 c---c/2
        self.branch4_1 = nn.Sequential(Conv2d( 1, 128, 16, same_padding=True, bn=True),
                                     )

        self.branch4_2 = nn.Sequential(Conv2d( 1, 128, 64, same_padding=True, bn=True),
                                     Conv2d( 3, 64, 16, same_padding=True, bn=True))

        self.branch4_3 = nn.Sequential(Conv2d( 1, 128, 64, same_padding=True, bn=True),
                                     Conv2d( 5, 64, 16, same_padding=True, bn=True))

        self.branch4_4 = nn.Sequential(Conv2d( 1, 128, 64, same_padding=True, bn=True),
                                     Conv2d( 7, 64, 16, same_padding=True, bn=True))

        # transposed conv
        # self.Transposed_9 = nn.Sequential(Trans_Conv2d( 64, 64, 9, stride=(2,2), padding=(4,4),output_padding=(1,1),bn=True)
        #                              )
        # self.Transposed_7 = nn.Sequential(Trans_Conv2d( 64, 32, 7, stride=(2,2), padding=(3,3),output_padding=(1,1),bn=True)
        #                              )
        # self.Transposed_5 = nn.Sequential(Trans_Conv2d( 32, 16, 5, stride=(2,2), padding=(2,2),output_padding=(1,1),bn=True)
        #                              )

        self.Transposed_9 = nn.Sequential(Conv2d( 9, 64, 64, same_padding=True, bn=True),
        nn.ConvTranspose2d( 64, 64, 2,2),
        nn.InstanceNorm2d(64)
                                     )
        self.Transposed_7 = nn.Sequential(Conv2d( 7, 64, 32, same_padding=True, bn=True),
        nn.ConvTranspose2d( 32, 32, 2,2),
        nn.InstanceNorm2d(32)
                                     )
        self.Transposed_5 = nn.Sequential(Conv2d( 5, 32, 16, same_padding=True, bn=True),
        nn.ConvTranspose2d( 16, 16, 2,2),
        nn.InstanceNorm2d(16)
                                     )
        # last two conv
        self.conv3_16 = nn.Sequential(Conv2d( 3, 16, 16, same_padding=True, bn=True)
                                     )
        self.conv5_16 = nn.Sequential(Conv2d( 5, 16, 16, same_padding=True, bn=True)
                                     )



        self.fuse = nn.Sequential(Conv2d( 1, 16, 1, same_padding=True, bn=True))

    def FME(self, im_data):
        # pdb.set_trace()
        x1_1 = self.branch1_1(im_data)
        # 2x16x256x256
        x1_2 = self.branch1_2(im_data)
        x1_3 = self.branch1_3(im_data)
        x1_4 = self.branch1_4(im_data)
        x1 = torch.cat((x1_1,x1_2,x1_3,x1_4),1)
        # 2x64x256x256


        # pdb.set_trace()
        x2_1 = self.branch2_1(x1)
        # 2x32x128x128
        x2_2 = self.branch2_2(x1)
        x2_3 = self.branch2_3(x1)
        x2_4 = self.branch2_4(x1)
        x2 = torch.cat((x2_1,x2_2,x2_3,x2_4),1)
        # 2x128x128x128


        # pdb.set_trace()
        x3_1 = self.branch3_1(x2)
        # 2x32x64x64
        x3_2 = self.branch3_2(x2)
        x3_3 = self.branch3_3(x2)
        x3_4 = self.branch3_4(x2)
        x3 = torch.cat((x3_1,x3_2,x3_3,x3_4),1)
        # 2x128x64x64

        # pdb.set_trace()
        x4_1 = self.branch4_1(x3)
        # 2x16x32x32
        x4_2 = self.branch4_2(x3)
        x4_3 = self.branch4_3(x3)
        x4_4 = self.branch4_4(x3)
        x4 = torch.cat((x4_1,x4_2,x4_3,x4_4),1)
        # 2x64x64x64


        return x4

    def DME(self, x):
        # pdb.set_trace()
        # 2x64x64x64
        x = self.Transposed_9(x)
        x = self.Transposed_7(x)
        x = self.Transposed_5(x)
        # 2x16x512x512

        return x

    def forward(self, x):
        # pdb.set_trace()
        x = self.FME(x)
        x = self.DME(x)

        x = self.conv3_16(x)
        x = self.conv5_16(x)


        x = self.fuse(x)


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
