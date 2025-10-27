"""
DenseNet Architecture (2017)
Dense Convolutional Networks
Authors: Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
"""

import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    """Single dense layer with batch norm and ReLU"""
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, 
                              padding=1, bias=False)
        
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)  # Dense connection
        return out

class DenseBlock(nn.Module):
    """Dense block with multiple dense layers"""
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate, growth_rate))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    """Transition layer between dense blocks"""
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    """
    DenseNet Architecture (DenseNet-121 by default)
    Input: 224x224 RGB images
    Output: 1000 classes (ImageNet) or configurable
    Key innovation: Dense connections - each layer connects to all previous layers
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_classes=1000, input_channels=3):
        super(DenseNet, self).__init__()
        
        # Initial convolution
        num_init_features = 2 * growth_rate
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, num_init_features, kernel_size=7, 
                     stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks
        self.features = nn.Sequential()
        num_features = num_init_features
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_features, growth_rate, num_layers)
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = num_features // 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def get_model_info(self):
        return {
            'name': 'DenseNet-121',
            'year': 2017,
            'parameters': sum(p.numel() for p in self.parameters()),
            'input_size': (224, 224),
            'authors': 'Gao Huang et al.',
            'key_features': 'Dense connections, feature reuse, gradient flow improvement'
        }

def create_densenet121(num_classes=1000, input_channels=3):
    """Factory function to create DenseNet-121 model"""
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                   num_classes=num_classes, input_channels=input_channels)