"""
MobileNetV2 uses inverted residual and linear bottleneck layers. This is because :
    1. Feature maps are able to be encoded in low dimensional subspaces
    2. Non-linear activations results in information loss although they increase complexity of representation

The operations are as below :

a. Stride 1 block :
- Input
- 1x1 Convolution with Relu6
- Depthwise Convolution with Relu6
- 1x1 Convolution 
- Add 

b. Stride 2 block :
- Input
- 1x1 Convolution with Relu6
- Depthwise Convolution with stride = 2 and Relu6
- 1x1 Convolution 

The point-wise 1x1 convolution after input layer brings the low-d input layer to higher-d space. Non-linear 
activations such as ReLU6 is suitable to be applied here. The expansion factor is t. The output is tk channels.
To carry out spatial filtering, depthwise convolution is performed and then ReLU6 is applied.
Then, point-wise 1x1 convolution is done to the output to project the feature maps down to lower-d. This causes
loss in information so no ReLU6 activation is added. If stride is 1 (input and output same shape), resnet 
connection is added. This helps in backpropagation of errors. 

To appease (2), ReLu6 is only applied on higher-d feature maps.
"""

import torch
import torch.nn as nn

class bottleneck(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout, stride, padding):
        super(bottleneck, self).__init__()
        self.stride = stride
        self.pointwise1 = nn.Conv2d(nin, nin*kernels_per_layer, kernel_size = 1, stride = 1)
        self.bn1 = nn.BatchNorm2d(nin*kernels_per_layer)
        self.relu1 = nn.ReLU6()
        self.depthwise1 = nn.Conv2d(nin*kernels_per_layer, nin*kernels_per_layer, kernel_size = 3, stride = stride, padding = padding, groups = nin*kernels_per_layer)
        self.bn2 = nn.BatchNorm2d(nin*kernels_per_layer)
        self.relu2 = nn.ReLU6()
        self.pointwise2 = nn.Conv2d(nin*kernels_per_layer, nout, kernel_size = 1, stride = 1)
        self.bn3 = nn.BatchNorm2d(nout)
        
    def forward(self, x):
        x_res = x
        # expansion to higher-d
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # spatial filtering
        x = self.depthwise1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # projection to lower-d
        x = self.pointwise2(x)
        x = self.bn3(x)
        
        #print('x :', x.shape)
        #print('x_res', x_res.shape)
        if  x.shape[1:] != x_res.shape[1:]: return x
        else: return x + x_res
        

class MobileNetV2(nn.Module):
    def __init__(self, in_channels, kernels_per_layer):
        super(MobileNetV2, self).__init__()
        self.seq1 = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size = 3, stride = 2, padding = 1),
                bottleneck(32, 1, 16, 1, 1),
                bottleneck(16, kernels_per_layer, 24, 2, 1),
                bottleneck(24, kernels_per_layer, 24, 1, 1),
                bottleneck(24, kernels_per_layer, 32, 2, 1),
                bottleneck(32, kernels_per_layer, 32, 1, 1),
                bottleneck(32, kernels_per_layer, 32, 1, 1),
                bottleneck(32, kernels_per_layer, 64, 2, 1),
                bottleneck(64, kernels_per_layer, 64, 1, 1),
                bottleneck(64, kernels_per_layer, 64, 1, 1),
                bottleneck(64, kernels_per_layer, 64, 1, 1),
                bottleneck(64, kernels_per_layer, 96, 1, 1),
                bottleneck(96, kernels_per_layer, 96, 1, 1),
                bottleneck(96, kernels_per_layer, 96, 1, 1),
                bottleneck(96, kernels_per_layer, 160, 2, 1),
                bottleneck(160, kernels_per_layer, 160, 1, 1),
                bottleneck(160, kernels_per_layer, 160, 1, 1),
                bottleneck(160, kernels_per_layer, 320, 1, 1),
                nn.Conv2d(320, 1280, kernel_size = 1, stride = 1),
                nn.AvgPool2d(kernel_size = 7),
                nn.Conv2d(1280, 1000, kernel_size = 1, stride = 1),
                nn.Softmax()
            )
        
    def forward(self, x):
        x=  self.seq1(x)
        return x.reshape(-1, 1000)
        
if __name__ == "__main__":
    device = 'cpu'
    x = torch.randn(10, 3, 224, 224).to(device)
    mn_model = MobileNetV2(3, 6) 
    #print(mn_model)
    print(mn_model(x).shape)
