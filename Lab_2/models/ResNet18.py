import torch
from torch import nn
from torch.nn import functional as F
class BasicBlock(nn.Module):
    def __init__(self, out_channels: int, stride: int = 1, use_1x1conv: bool = False, **kwargs):
        super().__init__(**kwargs) 
       
        self.conv_1 = nn.LazyConv2d(
            out_channels = out_channels,
            kernel_size = 3,
            stride = stride, # if stride = 2 (meaning the skip connection between different number of kernels), the feature map size is halved, the number of ﬁlters is doubled so as to preserve the time complexity per layer. We perform downsampling directly by convolutional layers that have a stride of 2 according to original paper.
            padding = 1
        )

        self.conv_2 = nn.LazyConv2d(
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )

        if use_1x1conv:
            self.shortcut_conv_3 = nn.LazyConv2d(
                out_channels=out_channels,
                kernel_size=1,
                stride = stride,
                padding=0,
            )
        else:
            self.shortcut_conv_3 = None
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.residual = nn.Sequential(
            self.conv_1,
            self.bn_1,
            nn.ReLU(),
            self.conv_2,
            self.bn_2,
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.residual(x)
        if self.shortcut_conv_3:
            return F.relu(x_res + self.shortcut_conv_3(x))
        return F.relu(x_res + x)
class ResNet18(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs) 
        self.conv1 = nn.LazyConv2d(
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
        )

        self.bn = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv_2x = nn.Sequential(
            BasicBlock(64),
            BasicBlock(64)
        )

        self.conv_3x = nn.Sequential(
            BasicBlock(128, stride=2, use_1x1conv=True),
            BasicBlock(128),
        )

        self.conv_4x = nn.Sequential(
            BasicBlock(256, stride=2, use_1x1conv=True),
            BasicBlock(256),
        )

        self.conv_5x = nn.Sequential(
            BasicBlock(512, stride=2, use_1x1conv=True),
            BasicBlock(512),
        )
        
        self.avg_pool = nn.AvgPool2d( # (bs, 512, 1, 1)
            kernel_size=7,
            stride=1,
            padding=0
        )

        self.fc = nn.Linear(
            in_features=512,
            out_features=num_classes    
        )

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) # (bs, 64, 112, 112)
        x = F.relu(self.bn(x))
        x = self.max_pool(x) # 
        x = self.conv_2x(x) # (bs, 64, 56, 56)
        x = self.conv_3x(x) # (bs, 128, 28, 28)
        x = self.conv_4x(x) # (bs, 256, 14, 14)
        x = self.conv_5x(x) # (bs, 512, 7, 7)
        x = self.avg_pool(x) # (bs, 512, 1, 1)
        x = x.view(x.size(0), -1) # (bs, 512)
        x = self.fc(x) # (bs, num_classes)
        return x

'''
Option 2 for more concise code

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
'''