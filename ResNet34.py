import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, kernel_size = 3, padding = 1, bias = False):
        super(Block, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
                nn.BatchNorm2d(out_channels),
            )

        if stride != 1 or in_channels != out_channels :
            
            self.skip_connect = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0, bias = False),
                nn.BatchNorm2d(out_channels),        
                )
            
        else : self.skip_connect = nn.Sequential()
        
        self.relu_final = nn.ReLU()
        
    def forward(self, x):
        x_skip = x
        x = self.layer1(x)
        x = self.layer2(x)
        x += self.skip_connect(x_skip)
        x = self.relu_final(x)
        return x


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.block1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        
        self.block2 = nn.Sequential(
                nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
                Block(64, 64),
                Block(64, 64),
                Block(64, 64),
            )
        
        self.block3 = nn.Sequential(
                Block(64, 128, 2),
                Block(128, 128),
                Block(128, 128),
                Block(128, 128),
            )
        
        self.block4 = nn.Sequential(
                Block(128, 256),
                Block(256, 256),
                Block(256, 256),
                Block(256, 256),
                Block(256, 256),
                Block(256, 256),
            )
        
        self.block5 = nn.Sequential(
                Block(256, 512, 2),
                Block(512, 512),
                Block(512, 512),
            )
        
        self.avg_pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512*7*7, 1000)
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.block1(x)
        #print(x.shape)
        x = self.block2(x)
        #print(x.shape)
        x = self.block3(x)
        #print(x.shape)
        x = self.block4(x)
        #print(x.shape)
        x = self.block5(x)
        #print(x.shape)
        x = self.avg_pool(x)
        #print(x.shape)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return self.softmax(x)
        
     
    
if __name__ == "__main__":
    device = 'cpu'
    resnet_model = ResNet34()
    #print(resnet_model)
    x = torch.randn(10, 3, 224, 224).to(device)
    print(resnet_model(x).shape)

