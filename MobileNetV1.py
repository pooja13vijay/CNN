import torch
import torch.nn as nn
import numpy as np

class depthwise_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout, stride, padding):
        super(depthwise_conv, self).__init__()
        #print(round(nin/ kernels_per_layer))
        self.depthwise = nn.Conv2d(nin, nin*kernels_per_layer, kernel_size = 3, stride = stride, padding = padding, groups = round(nin/ kernels_per_layer))
        self.bn1 = nn.BatchNorm2d(nin*kernels_per_layer)
        self.relu1 = nn.ReLU()
        self.pointwise = nn.Conv2d(nin*kernels_per_layer, nout, kernel_size = 1)
        self.bn2 = nn.BatchNorm2d(nout)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x= self.pointwise(x)
        x = self.bn2(x)
        return self.relu2(x)
    
class MobileNet(nn.Module):
    def __init__(self, in_channels, kernels_per_layer):
        super(MobileNet, self).__init__()
        self.seq = nn.Sequential(
            self.block(in_channels, 32, 3, 2, 1),
            depthwise_conv(32, kernels_per_layer, 32, 1, 1),
            self.block(32, 64, 1, 1, 0),
            depthwise_conv(64, kernels_per_layer, 64, 2, 1),
            self.block(64, 128, 1, 1, 0),
            depthwise_conv(128, kernels_per_layer, 128, 1, 1),
            self.block(128, 128, 1, 1, 0),
            depthwise_conv(128, kernels_per_layer, 128, 2, 1),
            self.block(128, 256, 1, 1, 0),
            depthwise_conv(256, kernels_per_layer, 256, 1, 1),
            self.block(256, 256, 1, 1, 0),
            depthwise_conv(256, kernels_per_layer, 256, 2, 1),
            self.block(256, 512, 1, 1, 0),
            depthwise_conv(512, kernels_per_layer, 512, 1, 1),
            self.block(512, 512, 1, 1, 0),
            depthwise_conv(512, kernels_per_layer, 512, 1, 1),
            self.block(512, 512, 1, 1, 0),
            depthwise_conv(512, kernels_per_layer, 512, 1, 1),
            self.block(512, 512, 1, 1, 0),
            depthwise_conv(512, kernels_per_layer, 512, 1, 1),
            self.block(512, 512, 1, 1, 0),
            depthwise_conv(512, kernels_per_layer, 512, 1, 1),
            self.block(512, 512, 1, 1, 0),
            depthwise_conv(512, kernels_per_layer, 512, 2, 1),
            self.block(512, 1024, 1, 1, 0),
            depthwise_conv(1024, kernels_per_layer, 1024, 2, 4),
            self.block(1024, 1024, 1, 1, 0),
            nn.AvgPool2d(kernel_size = 7, stride = 1),
            nn.Flatten(),
            nn.Linear(1024, 1000),
            nn.Softmax()
            )
                
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
    
    def forward(self, x):
        return self.seq(x)
        

if __name__ == "__main__":
    device = 'cpu'
    x = torch.randn(10, 3, 224, 224).to(device)
    mn_model = MobileNet(3, 1) # (3, 2) for 0.5x
    #print(mn_model)
    print(mn_model(x).shape)



































































