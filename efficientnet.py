"""
EfficientNet Architecture (2019)
Efficient network scaling
Authors: Mingxing Tan and Quoc V. Le (Google)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SwishActivation(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block"""
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, se_channels, 1),
            SwishActivation(),
            nn.Conv2d(se_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x * self.se(x)

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SwishActivation()
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=stride,
                     padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            SwishActivation()
        ])
        
        # SE block
        layers.append(SEBlock(hidden_dim))
        
        # Projection phase
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)

class EfficientNet(nn.Module):
    """
    EfficientNet-B0 Architecture
    Input: 224x224 RGB images
    Output: 1000 classes (ImageNet) or configurable
    Key innovation: Compound scaling (depth, width, resolution)
    """
    def __init__(self, num_classes=1000, input_channels=3):
        super(EfficientNet, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            SwishActivation()
        )
        
        # Building blocks (simplified EfficientNet-B0)
        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 3, 1, 1),
            MBConvBlock(16, 24, 3, 2, 6),
            MBConvBlock(24, 24, 3, 1, 6),
            MBConvBlock(24, 40, 5, 2, 6),
            MBConvBlock(40, 40, 5, 1, 6),
            MBConvBlock(40, 80, 3, 2, 6),
            MBConvBlock(80, 80, 3, 1, 6),
            MBConvBlock(80, 80, 3, 1, 6),
            MBConvBlock(80, 112, 5, 1, 6),
            MBConvBlock(112, 112, 5, 1, 6),
            MBConvBlock(112, 112, 5, 1, 6),
            MBConvBlock(112, 192, 5, 2, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 192, 5, 1, 6),
            MBConvBlock(192, 320, 3, 1, 6),
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            SwishActivation()
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def get_model_info(self):
        return {
            'name': 'EfficientNet-B0',
            'year': 2019,
            'parameters': sum(p.numel() for p in self.parameters()),
            'input_size': (224, 224),
            'authors': 'Mingxing Tan and Quoc V. Le',
            'key_features': 'Compound scaling, mobile inverted bottleneck, SE blocks, Swish'
        }

def create_efficientnet_b0(num_classes=1000, input_channels=3):
    """Factory function to create EfficientNet-B0 model"""
    return EfficientNet(num_classes=num_classes, input_channels=input_channels)