import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class AlexNet(nn.Module):
    def __init__(self, in_channels = 3):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 96, kernel_size = 11, stride = 4)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1)
        self.relu5 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.linear1 = nn.Linear(256*6*6, 4096)
        self.relu6 = nn.ReLU()
        self.linear2 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.linear3 = nn.Linear(4096, 1000)
        self.softmax1 = nn.Softmax()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool3(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu6(x)
        x = self.linear2(x)
        x = self.relu7(x)
        x = self.linear3(x)
        return self.softmax1(x)
    
if __name__ == "__main__":
    device = 'cpu'
    alexnet_model = AlexNet()
    #print(vgg_model)
    x = torch.randn(10, 3, 227, 227).to(device)
    print(alexnet_model(x).shape)

