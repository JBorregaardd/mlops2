import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(2)                               
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.pool2 = nn.MaxPool2d(2)                              
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)