import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #First Conv's
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.norm1=nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.norm2=nn.BatchNorm2d(16)
        self.conv3=nn.Conv2d(16,32,5)
        self.norm3=nn.BatchNorm2d(32)
        #Second set of Conv's
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4=nn.Conv2d(32,64,3)
        self.norm4=nn.BatchNorm2d(64)
        self.conv5=nn.Conv2d(64,128,3)
        self.norm5=nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #First Convs
        x = self.norm1(F.relu(self.conv1(x)))
        x = self.norm2(F.relu(self.conv2(x)))
        #Second set of Conv's
        x = self.norm3(self.pool(F.relu(self.conv3(x))))
        x = self.norm4(self.pool(F.relu(self.conv4(x))))
        x = self.norm5(self.pool(F.relu(self.conv5(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
