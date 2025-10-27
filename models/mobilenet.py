"""
MobileNet Architecture (2017)
Efficient CNN for mobile and embedded devices
Authors: Andrew G. Howard et al. (Google)
"""

import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # Pointwise convolution
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MobileNet(nn.Module):
    """
    MobileNet Architecture
    Input: 224x224 RGB images
    Output: 1000 classes (ImageNet) or configurable
    Parameters: ~4.2 million (very lightweight!)
    Key innovation: Depthwise separable convolutions for efficiency
    """
    def __init__(self, num_classes=1000, input_channels=3, width_multiplier=1.0):
        super(MobileNet, self).__init__()
        
        # Standard convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, int(32 * width_multiplier), kernel_size=3,
                     stride=2, padding=1, bias=False),
            nn.BatchNorm2d(int(32 * width_multiplier)),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise separable convolutions
        self.features = nn.Sequential(
            DepthwiseSeparableConv(int(32 * width_multiplier), int(64 * width_multiplier), stride=1),
            DepthwiseSeparableConv(int(64 * width_multiplier), int(128 * width_multiplier), stride=2),
            DepthwiseSeparableConv(int(128 * width_multiplier), int(128 * width_multiplier), stride=1),
            DepthwiseSeparableConv(int(128 * width_multiplier), int(256 * width_multiplier), stride=2),
            DepthwiseSeparableConv(int(256 * width_multiplier), int(256 * width_multiplier), stride=1),
            DepthwiseSeparableConv(int(256 * width_multiplier), int(512 * width_multiplier), stride=2),
            
            # 5x 512 filters
            DepthwiseSeparableConv(int(512 * width_multiplier), int(512 * width_multiplier), stride=1),
            DepthwiseSeparableConv(int(512 * width_multiplier), int(512 * width_multiplier), stride=1),
            DepthwiseSeparableConv(int(512 * width_multiplier), int(512 * width_multiplier), stride=1),
            DepthwiseSeparableConv(int(512 * width_multiplier), int(512 * width_multiplier), stride=1),
            DepthwiseSeparableConv(int(512 * width_multiplier), int(512 * width_multiplier), stride=1),
            
            DepthwiseSeparableConv(int(512 * width_multiplier), int(1024 * width_multiplier), stride=2),
            DepthwiseSeparableConv(int(1024 * width_multiplier), int(1024 * width_multiplier), stride=1),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * width_multiplier), num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_model_info(self):
        return {
            'name': 'MobileNet-v1',
            'year': 2017,
            'parameters': sum(p.numel() for p in self.parameters()),
            'input_size': (224, 224),
            'authors': 'Andrew G. Howard et al.',
            'key_features': 'Depthwise separable convolutions, mobile-optimized, lightweight'
        }

def create_mobilenet(num_classes=1000, input_channels=3, width_multiplier=1.0):
    """Factory function to create MobileNet model"""
    return MobileNet(num_classes=num_classes, input_channels=input_channels, 
                    width_multiplier=width_multiplier)