import numpy as np
import torch
import torchvision
import torch.nn as nn


############################# Create UNet Model ################################
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,  out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.LeakyReLU(inplace=True)
    )   

def double_conv_final(in_channels, out_channels1, out_channels2, out_channels3):
    return nn.Sequential(
        nn.Conv2d(in_channels,   out_channels1, 1, padding=0),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels1, out_channels2, 1, padding=0),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels2, out_channels3, 1, padding=0),
        nn.LeakyReLU(inplace=True)
    )

class UNet2(nn.Module):

    def __init__(self):
        super(UNet2,self).__init__()
                
        self.dconv_down1 = double_conv(2, 4)
        self.dconv_down2 = double_conv(4, 8)
        self.dconv_down3 = double_conv(8, 16)
        self.dconv_down4 = double_conv(16, 32)        

        self.maxpool = nn.MaxPool2d(2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', 
                                    align_corners=True)        
        
        self.dconv_up3 = double_conv(32 + 16, 32)     #concatenated with conv_down3 
        self.dconv_up2 = double_conv(32 + 8, 16)      #concatenated with conv_down2
        self.dconv_up1 = double_conv(16 + 2 + 4, 8)   #concatenated with conv_down1 and 2 inputs
        self.conv_last = nn.Conv2d(8, 1, 1) 
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x1    = self.maxpool(conv1)
        
        conv2 = self.dconv_down2(x1)
        x2    = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x2)
        x3    = self.maxpool(conv3)   
        
        x4 = self.dconv_down4(x3)
        
        x5 = self.upsample(x4)  
        x6 = torch.cat([x5, conv3], dim=1)
        x7 = self.dconv_up3(x6)

        x8 = self.upsample(x7)        
        x9 = torch.cat([x8, conv2], dim=1)       
        x10 = self.dconv_up2(x9)

        x11 = self.upsample(x10)        
        x12 = torch.cat([x11, conv1, x], dim=1)           
        x13 = self.dconv_up1(x12)
        out = self.conv_last(x13)
        
        return out
