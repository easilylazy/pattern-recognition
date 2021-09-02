import torch
import torch.nn as nn
import torch.nn.functional as F
unit_num=2
class ResNet(nn.Module):
    def ResUnit(self,bn,layer,xi):
        x = F.relu(layer(xi))
        x = F.relu(layer(x))
        xo=xi+x
        return xo
    def __init__(self):
        super(ResNet, self).__init__()
        # 1 input image channel, 6 output channels, 7x7 square convolution
        # kernel
        self.conv1_0 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv2_0 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3_0 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.bn1=nn.BatchNorm2d(16)
        self.bn2=nn.BatchNorm2d(32)
        self.bn3=nn.BatchNorm2d(64)
        self.fc = nn.Linear(64,10)  

    def forward(self, x):
        # dimension 16
        x = F.relu(self.conv1_0(x))
        x = self.ResUnit(self.bn1,self.conv1,x)
        for i in range(unit_num):
            x = self.ResUnit(self.bn1,self.conv1,x)

        # x2_0=x1_2+x
        x = F.relu(self.conv2_0(x))
        # dimension 32
        x = F.relu(self.conv2(x))
        for i in range(unit_num):
            x = self.ResUnit(self.bn2,self.conv2,x)

        x = F.relu(self.conv3_0(x))
        # dimension 64
        x = F.relu(self.conv3(x))
        for i in range(unit_num):
            x = self.ResUnit(self.bn3,self.conv3,x)

        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.fc(x)
        return x

class ResNet_BN(nn.Module):
    def ResUnit(self,bn,layer,xi):
        x = F.relu(bn(layer(xi)))
        x = F.relu(bn(layer(x)))
        xo=xi+x
        return xo
    def __init__(self):
        super(ResNet_BN, self).__init__()
        # 1 input image channel, 6 output channels, 7x7 square convolution
        # kernel
        self.conv1_0 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv2_0 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3_0 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.bn1=nn.BatchNorm2d(16)
        self.bn2=nn.BatchNorm2d(32)
        self.bn3=nn.BatchNorm2d(64)
        self.fc = nn.Linear(64,10)  

    def forward(self, x):
        # dimension 16
        x = F.relu(self.bn1(self.conv1_0(x)))
        x = self.ResUnit(self.bn1,self.conv1,x)
        for i in range(unit_num):
            x = self.ResUnit(self.bn1,self.conv1,x)

        # x2_0=x1_2+x
        x = F.relu(self.bn2(self.conv2_0(x)))
        # dimension 32
        x = F.relu(self.bn2(self.conv2(x)))
        for i in range(unit_num):
            x = self.ResUnit(self.bn2,self.conv2,x)

        x = F.relu(self.bn3(self.conv3_0(x)))
        # dimension 64
        x = F.relu(self.bn3(self.conv3(x)))
        for i in range(unit_num):
            x = self.ResUnit(self.bn3,self.conv3,x)

        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.fc(x)
        return x