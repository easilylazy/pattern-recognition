import torch
import torch.nn as nn
import torch.nn.functional as F

layer_num = [18, 34, 50, 101, 152]
# unit num of each layer
num_config = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
}


class ResUnit_BN(nn.Module):
    def __init__(self, channels, layers=2):
        super(ResUnit_BN, self).__init__()
        self.layers = layers
        self.relu = nn.ReLU(inplace=True)
        self.convList = nn.ModuleList()
        self.bnList = nn.ModuleList()
        for i in range(layers):
            conv = nn.Conv2d(channels, channels, 3, 1, 1)
            self.convList.append(conv)
            bn = nn.BatchNorm2d(channels)
            self.bnList.append(bn)

    def forward(self, xi):
        x = self.relu(self.bnList[0](self.convList[0](xi)))
        x = self.relu(self.bnList[1](self.convList[1](x)))
        xo = xi + x
        xo = self.relu(xo)
        return xo


class DimUnit_BN(nn.Module):
    """
    dimension change
    """

    def __init__(self, in_channels, out_channels, stride=2, option="B"):
        super(DimUnit_BN, self).__init__()
        self.option = option
        self.relu = nn.ReLU(inplace=True)
        self.convList = nn.ModuleList()
        self.bnList = nn.ModuleList()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, 1, stride, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn1_2 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pad = nn.ConstantPad3d((0, 0, 0, 0, 0, out_channels - in_channels), 0)

    def forward(self, xi):

        x = self.relu(self.bn1(self.conv1(xi)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.option == "A":
            xo = self.pad(xi[:, :, ::2, ::2]) + x
        elif self.option == "B":
            x1_2 = self.bn1_2(self.conv1_2(xi))
            xo = x1_2 + x
        else:
            xo = x
        xo = self.relu(xo)
        return xo


class ResNet_BN(nn.Module):
    def __init__(self, total_layer=18):
        super(ResNet_BN, self).__init__()
        self.total_layer = total_layer
        self.relu = nn.ReLU(inplace=True)
        # 1 input image channel, 6 output channels, 7x7 square convolution
        # kernel
        self.conv1_0 = nn.Conv2d(3, 64, 7, 2, 3)
        self.res1 = ResUnit_BN(16)
        self.pool1 = nn.MaxPool2d((3, 3), 2, 1)

        self.resunits = nn.ModuleList()
        channelsList = [64, 128, 256, 512]
        self.config_layer = num_config[total_layer]
        for channels, unit_num in zip(channelsList, self.config_layer):
            # connect first way
            for _ in range(unit_num - 1):
                resunit = ResUnit_BN(channels)
                self.resunits.append(resunit)
        self.DimUnit_BN1 = ResUnit_BN(64)
        self.DimUnit_BN2 = DimUnit_BN(64, 128)
        self.DimUnit_BN3 = DimUnit_BN(128, 256)
        self.DimUnit_BN4 = DimUnit_BN(256, 512)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(512, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        # dimension 64
        x = self.relu(self.bn1(self.pool1(self.conv1_0(x))))
        x = self.DimUnit_BN1(x)
        layer = 0
        for i in range(self.config_layer[0] - 1):
            x = self.resunits[layer](x)
            layer += 1
        # dimension 128
        x = self.DimUnit_BN2(x)
        for i in range(self.config_layer[1] - 1):
            x = self.resunits[layer](x)
            layer += 1
        # dimension 256
        x = self.DimUnit_BN3(x)
        for i in range(self.config_layer[2] - 1):
            x = self.resunits[layer](x)
            layer += 1
        # dimension 512
        x = self.DimUnit_BN4(x)
        for i in range(self.config_layer[3] - 1):
            x = self.resunits[layer](x)
            layer += 1
        x = self.pool(x)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = self.fc(x)
        return x

