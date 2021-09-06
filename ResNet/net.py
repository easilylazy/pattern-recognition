import torch
import torch.nn as nn
import torch.nn.functional as F
unit_num=2
class DimUnit(nn.Module):
    '''
    dimension change
    '''
    def __init__(self,in_channels, out_channels,stride=2):
        super(DimUnit,self).__init__()
        self.convList=nn.ModuleList()
        self.bnList=nn.ModuleList()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

    def forward(self,x):
        x = F.relu((self.conv1(x)))
        x = F.relu((self.conv2(x)))
        return x
class ResUnit(nn.Module):
    def __init__(self,channels,layers=2):
        super(ResUnit,self).__init__()
        self.layers=layers
        self.convList=nn.ModuleList()
        for i in range(layers):
            conv = nn.Conv2d(channels, channels, 3, 1, 1)
            self.convList.append(conv)


    def forward(self,xi):
        x = F.relu(self.convList[0](xi))
        x = F.relu(self.convList[1](x))
        xo=xi+x
        return xo
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # 1 input image channel, 6 output channels, 7x7 square convolution
        # kernel
        self.conv1_0 = nn.Conv2d(3, 16, 3, 1, 1)
        self.res1=ResUnit(16)

        self.resunits=nn.ModuleList()
        channelsList=[16,32,64]
        for channels in channelsList:
            for i in range(unit_num):
                resunit=ResUnit(channels)
                self.resunits.append(resunit)
        self.DimUnit2=DimUnit(16,32)
        self.DimUnit3=DimUnit(32,64)

        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,10)  

    def forward(self, x):
        # dimension 16
        x = F.relu((self.conv1_0(x)))
        x = self.res1(x)
        for i in range(unit_num):
            x = self.resunits[i](x)

        # # dimension 32
        x = self.DimUnit2(x)

        for i in range(unit_num):
            x = self.resunits[unit_num+i](x)

        # # dimension 64
        x = self.DimUnit3(x)
        for i in range(unit_num):
            x = self.resunits[unit_num*2+i](x)

        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.fc(x)
        return x
class ResUnit_BN(nn.Module):
    def __init__(self,channels,layers=2):
        super(ResUnit_BN,self).__init__()
        self.layers=layers
        self.convList=nn.ModuleList()
        self.bnList=nn.ModuleList()
        for i in range(layers):
            conv = nn.Conv2d(channels, channels, 3, 1, 1)
            self.convList.append(conv)
            bn=nn.BatchNorm2d(channels)
            self.bnList.append(bn)


    def forward(self,xi):
        x = F.relu(self.bnList[0](self.convList[0](xi)))
        x = F.relu(self.bnList[1](self.convList[1](x)))
        xo=xi+x
        xo=F.relu(xo)
        return xo
class DimUnit_BN(nn.Module):
    '''
    dimension change
    '''
    def __init__(self,in_channels, out_channels,stride=2,diff=8,option='B'):
        super(DimUnit_BN,self).__init__()
        self.option=option
        self.convList=nn.ModuleList()
        self.bnList=nn.ModuleList()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.bn1_2=nn.BatchNorm2d(out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.pad=nn.ConstantPad3d((0,0,0,0,0,out_channels-in_channels),0)
        

    def forward(self,xi):
        
        x = F.relu(self.bn1(self.conv1(xi)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.option=='A':
            xo=self.pad(xi[:,:,::2,::2])+x
        elif self.option=='B':
            x1_2=self.bn1_2(self.conv1_2(xi))
            xo=x1_2+x
        else:
            xo=x
        xo=F.relu(xo)
        return xo
class ResNet_BN(nn.Module):
    def __init__(self,unit_num=2):
        super(ResNet_BN, self).__init__()
        self.unit_num=unit_num
        # 1 input image channel, 6 output channels, 7x7 square convolution
        # kernel
        self.conv1_0 = nn.Conv2d(3, 16, 3, 1, 1)
        self.res1=ResUnit_BN(16)

        self.resunits=nn.ModuleList()
        channelsList=[16,32,64]
        for channels in channelsList:
            for i in range(self.unit_num):
                resunit=ResUnit_BN(channels)
                self.resunits.append(resunit)
        self.DimUnit_BN2=DimUnit_BN(16,32)
        self.DimUnit_BN3=DimUnit_BN(32,64)

        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.bn1=nn.BatchNorm2d(16)
        self.fc = nn.Linear(64,10)  

    def forward(self, x):
        # dimension 16
        x = F.relu(self.bn1(self.conv1_0(x)))
        x = self.res1(x)
        for i in range(self.unit_num):
            x = self.resunits[i](x)

        # # dimension 32
        x = self.DimUnit_BN2(x)

        for i in range(self.unit_num):
            x = self.resunits[self.unit_num+i](x)

        # # dimension 64
        x = self.DimUnit_BN3(x)
        for i in range(self.unit_num):
            x = self.resunits[self.unit_num*2+i](x)

        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.fc(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 7x7 square convolution
        # kernel
        self.conv1_0 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.conv2_0 = nn.Conv2d(16, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3_0 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)

        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64,10)  

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x1_0 = F.relu(self.conv1_0(x))
        x = F.relu(self.conv1(x1_0))
        x = F.relu(self.conv1(x))
        x1_1=x1_0+x
        x = F.relu(self.conv1(x1_1))
        x = F.relu(self.conv1(x))
        x1_2=x1_1+x
        x = F.relu(self.conv1(x1_2))
        x = F.relu(self.conv1(x))

        x2_0=x1_2+x
        x = F.relu(self.conv2_0(x2_0))
        x2_1 = F.relu(self.conv2(x))
        # dimension 32
        x = F.relu(self.conv2(x2_1))
        x = F.relu(self.conv2(x))
        x2_2=x2_1+x
        x = F.relu(self.conv2(x2_2))
        x = F.relu(self.conv2(x))

        x3_0=x2_2+x
        x = F.relu(self.conv3_0(x3_0))
        x3_1 = F.relu(self.conv3(x))
        # dimension 64
        x = F.relu(self.conv3(x3_1))
        x = F.relu(self.conv3(x))
        x3_2=x3_1+x
        x = F.relu(self.conv3(x3_2))
        x = F.relu(self.conv3(x))

        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.fc(x)
        return x
class ResUnit_BN_plain(nn.Module):
    def __init__(self,channels,layers=2):
        super(ResUnit_BN_plain,self).__init__()
        self.layers=layers
        self.convList=nn.ModuleList()
        self.bnList=nn.ModuleList()
        for i in range(layers):
            conv = nn.Conv2d(channels, channels, 3, 1, 1)
            self.convList.append(conv)
            bn=nn.BatchNorm2d(channels)
            self.bnList.append(bn)


    def forward(self,xi):
        x = F.relu(self.bnList[0](self.convList[0](xi)))
        x = F.relu(self.bnList[1](self.convList[1](x)))
        x = F.relu(x)
        return x
class DimUnit_BN_plain(nn.Module):
    '''
    dimension change
    '''
    def __init__(self,in_channels, out_channels,stride=2,diff=8):
        super(DimUnit_BN_plain,self).__init__()
        self.convList=nn.ModuleList()
        self.bnList=nn.ModuleList()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.pad=nn.ConstantPad3d((0,0,0,0,0,out_channels-in_channels),0)
        

    def forward(self,xi):
        x = F.relu(self.bn1(self.conv1(xi)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(x)
        return x
class ResNet_BN_plain(nn.Module):
    def __init__(self,unit_num=2):
        super(ResNet_BN_plain, self).__init__()
        self.unit_num=unit_num
        # 1 input image channel, 6 output channels, 7x7 square convolution
        # kernel
        self.conv1_0 = nn.Conv2d(3, 16, 3, 1, 1)
        self.res1=ResUnit_BN_plain(16)

        self.resunits=nn.ModuleList()
        channelsList=[16,32,64]
        for channels in channelsList:
            for i in range(self.unit_num):
                resunit=ResUnit_BN_plain(channels)
                self.resunits.append(resunit)
        self.DimUnit_BN2=DimUnit_BN_plain(16,32)
        self.DimUnit_BN3=DimUnit_BN_plain(32,64)

        self.pool=nn.AdaptiveAvgPool2d((1,1))
        self.bn1=nn.BatchNorm2d(16)
        self.fc = nn.Linear(64,10)  

    def forward(self, x):
        # dimension 16
        x = F.relu(self.bn1(self.conv1_0(x)))
        x = self.res1(x)
        for i in range(self.unit_num):
            x = self.resunits[i](x)

        # # dimension 32
        x = self.DimUnit_BN2(x)

        for i in range(self.unit_num):
            x = self.resunits[self.unit_num+i](x)

        # # dimension 64
        x = self.DimUnit_BN3(x)
        for i in range(self.unit_num):
            x = self.resunits[self.unit_num*2+i](x)

        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.fc(x)
        return x