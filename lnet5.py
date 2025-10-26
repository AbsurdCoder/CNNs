"""
LeNet-5 Architecture (1998)
First successful CNN for handwritten digit recognition
Author: Yann LeCun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 Architecture
    Input: 32x32 grayscale images
    Output: 10 classes (digits 0-9)
    Parameters: ~60,000
    """
    def __init__(self, num_classes=10, input_channels=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        # Input: (N, 1, 32, 32)
        x = F.relu(self.conv1(x))  # (N, 6, 32, 32)
        x = F.avg_pool2d(x, 2)     # (N, 6, 16, 16)
        x = F.relu(self.conv2(x))  # (N, 16, 12, 12)
        x = F.avg_pool2d(x, 2)     # (N, 16, 6, 6)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_model_info(self):
        return {
            'name': 'LeNet-5',
            'year': 1998,
            'parameters': sum(p.numel() for p in self.parameters()),
            'input_size': (32, 32),
            'authors': 'Yann LeCun et al.',
            'key_features': 'First successful CNN, average pooling, tanh activation'
        }

def create_lenet(num_classes=10, input_channels=1):
    """Factory function to create LeNet-5 model"""
    return LeNet5(num_classes=num_classes, input_channels=input_channels)