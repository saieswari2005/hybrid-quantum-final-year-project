import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """Convolutional Neural Network for image classification."""
    
    def __init__(self, num_classes=10, conv1_channels=16, conv2_channels=32, 
                 fc_hidden=128, dropout=0.5):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After 2 pooling layers: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(conv2_channels * 7 * 7, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
