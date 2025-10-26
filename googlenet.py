"""
GoogLeNet/Inception Architecture (2014)
ILSVRC 2014 Winner (classification task)
Authors: Christian Szegedy et al. (Google Research)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    """Inception module with parallel convolutions"""
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, out_pool):
        super(InceptionModule, self).__init__()
        
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 -> 5x5 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), 
                         self.branch3(x), self.branch4(x)], 1)

class GoogLeNet(nn.Module):
    """
    GoogLeNet Architecture
    Input: 224x224 RGB images
    Output: 1000 classes (ImageNet) or configurable
    Parameters: ~6 million (10x fewer than AlexNet!)
    Key innovation: Inception modules for multi-scale feature extraction
    """
    def __init__(self, num_classes=1000, input_channels=3):
        super(GoogLeNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Inception blocks
        self.inception3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def get_model_info(self):
        return {
            'name': 'GoogLeNet/Inception-v1',
            'year': 2014,
            'parameters': sum(p.numel() for p in self.parameters()),
            'input_size': (224, 224),
            'authors': 'Christian Szegedy et al.',
            'key_features': 'Inception modules, parallel multi-scale convolutions, efficient parameters'
        }

def create_googlenet(num_classes=1000, input_channels=3):
    """Factory function to create GoogLeNet model"""
    return GoogLeNet(num_classes=num_classes, input_channels=input_channels)