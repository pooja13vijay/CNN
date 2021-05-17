import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from torchvision.utils import make_grid

class VGG(nn.Module):
    def __init__(self, in_channels = 3, hidden_channels = 64):
        super(VGG, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size = 3, stride = 1, padding = 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size = 3, stride = 1, padding = 1)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size = 3, stride = 1, padding = 1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(hidden_channels*2, hidden_channels*2, kernel_size = 3, stride = 1, padding = 1)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv5 = nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size = 3, stride = 1, padding = 1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(hidden_channels*4, hidden_channels*4, kernel_size = 3, stride = 1, padding = 1)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(hidden_channels*4, hidden_channels*4, kernel_size = 3, stride = 1, padding = 1)
        self.relu7 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv8 = nn.Conv2d(hidden_channels*4, hidden_channels*8, kernel_size = 3, stride = 1, padding = 1)
        self.relu8 = nn.ReLU()
        self.conv9 = nn.Conv2d(hidden_channels*8, hidden_channels*8, kernel_size = 3, stride = 1, padding = 1)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(hidden_channels*8, hidden_channels*8, kernel_size = 3, stride = 1, padding = 1)
        self.relu10 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv11 = nn.Conv2d(hidden_channels*8, hidden_channels*8, kernel_size = 3, stride = 1, padding = 1)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(hidden_channels*8, hidden_channels*8, kernel_size = 3, stride = 1, padding = 1)
        self.relu12 = nn.ReLU()
        self.conv13 = nn.Conv2d(hidden_channels*8, hidden_channels*8, kernel_size = 3, stride = 1, padding = 1)
        self.relu13 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.linear1 = nn.Linear(512*7*7, 4096)
        self.relu14 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(4096, 4096)
        self.relu15 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(4096, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)        
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.maxpool3(x)    
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        x = self.maxpool4(x)   
        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.conv13(x)
        x = self.relu13(x)
        x = self.maxpool5(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear1(x)
        x = self.relu14(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu15(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        return x


if __name__ == "__main__":
    device = 'cpu'
    vgg_model = VGG()
    #print(vgg_model)
    x = torch.randn(10, 3, 224, 224).to(device)
    print(vgg_model(x).shape)

