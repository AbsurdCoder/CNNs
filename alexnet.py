"""
AlexNet Architecture (2012)
ILSVRC 2012 Winner - 17% top-5 error rate
Authors: Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton
"""

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    AlexNet Architecture
    Input: 224x224 RGB images
    Output: 1000 classes (ImageNet) or configurable
    Parameters: ~60 million
    Key innovations: ReLU, Dropout, GPU training
    """
    def __init__(self, num_classes=1000, input_channels=3):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv1: (3, 224, 224) -> (96, 55, 55)
            nn.Conv2d(input_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> (96, 27, 27)
            
            # Conv2: -> (256, 27, 27)
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> (256, 13, 13)
            
            # Conv3: -> (384, 13, 13)
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: -> (384, 13, 13)
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: -> (256, 13, 13)
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # -> (256, 6, 6)
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_model_info(self):
        return {
            'name': 'AlexNet',
            'year': 2012,
            'parameters': sum(p.numel() for p in self.parameters()),
            'input_size': (224, 224),
            'authors': 'Krizhevsky, Sutskever, Hinton',
            'key_features': 'ReLU activation, Dropout, GPU training, Local Response Normalization'
        }

def create_alexnet(num_classes=1000, input_channels=3):
    """Factory function to create AlexNet model"""
    return AlexNet(num_classes=num_classes, input_channels=input_channels)