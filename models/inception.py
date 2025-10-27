"""
Inception-v3 Architecture (2015)
Improved Inception architecture
Authors: Christian Szegedy et al. (Google)
"""

import torch
import torch.nn as nn

class InceptionV3(nn.Module):
    """
    Inception-v3 Architecture (Simplified)
    Input: 299x299 RGB images
    Output: 1000 classes (ImageNet) or configurable
    Key improvements: Factorized convolutions, batch normalization
    """
    def __init__(self, num_classes=1000, input_channels=3):
        super(InceptionV3, self).__init__()
        
        # Stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 80, kernel_size=1, bias=False),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
            nn.Conv2d(80, 192, kernel_size=3, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Simplified inception blocks
        self.inception_a = nn.Sequential(
            self._make_inception_block(192, 64, 96, 128, 16, 32, 32),
            self._make_inception_block(256, 64, 96, 128, 16, 32, 64),
            self._make_inception_block(288, 64, 96, 128, 16, 32, 64),
        )
        
        # Reduction
        self.reduction = nn.Sequential(
            nn.Conv2d(288, 384, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.inception_b = nn.Sequential(
            self._make_inception_block(384, 192, 128, 192, 128, 192, 192),
            self._make_inception_block(768, 192, 160, 192, 160, 192, 192),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(768, num_classes)
        
    def _make_inception_block(self, in_ch, out_1x1, r_3x3, out_3x3, r_5x5, out_5x5, out_pool):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_1x1 + out_3x3 + out_5x5 + out_pool, 
                     kernel_size=1, bias=False),
            nn.BatchNorm2d(out_1x1 + out_3x3 + out_5x5 + out_pool),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception_a(x)
        x = self.reduction(x)
        x = self.inception_b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def get_model_info(self):
        return {
            'name': 'Inception-v3',
            'year': 2015,
            'parameters': sum(p.numel() for p in self.parameters()),
            'input_size': (299, 299),
            'authors': 'Christian Szegedy et al.',
            'key_features': 'Factorized convolutions, batch normalization, label smoothing'
        }

def create_inceptionv3(num_classes=1000, input_channels=3):
    """Factory function to create Inception-v3 model"""
    return InceptionV3(num_classes=num_classes, input_channels=input_channels)