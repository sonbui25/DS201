import torch
from torch import nn
from torch.nn import functional as F
class Inception(nn.Module):
    def __init__(self, in_channels: int, c1: int, c2: int, c3: int, c4: int, **kwargs):
        super().__init__(**kwargs) # All strides are 1 because the input size remains unchanged after passing through this Inception block
        # Branch 1
        self.b1_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c1, # Number of output channels for branch 1
            kernel_size=1, # Kernel size 1x1
            stride=1, # Stride 1 (stride directly affects the output size)
            padding=0 # No padding (padding affects the information taken at the edge of the image, adding padding helps retain information at the edge of the image)
        )
        # Branch 2
        self.b2_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c2[0],
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.b2_2 = nn.Conv2d(
            in_channels=c2[0],
            out_channels=c2[1],
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Branch 3
        self.b3_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c3[0],
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.b3_2 = nn.Conv2d(
            in_channels=c3[0],
            out_channels=c3[1],
            kernel_size=5,
            stride=1,
            padding=2
        )

        # Branch 4
        self.b4_1 = nn.MaxPool2d(
            kernel_size=3,
            stride=1,
            padding=1,
            # ceil_mode=True # Helps to keep the output size equal to the input size when not divisible, avoiding information loss at the edge of the image, will round up when not divisible
        )
        self.b4_2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=c4,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Branch 1
        x_b1 = F.relu(self.b1_1(x))

        # Branch 2
        x_b2_1 = self.b2_1(x)
        x_b2_2 = F.relu(self.b2_2(F.relu(x_b2_1)))

        # Branch 3
        x_b3_1 = self.b3_1(x)
        x_b3_2 = F.relu(self.b3_2(F.relu(x_b3_1)))

        # Branch 4
        x_b4_1 = self.b4_1(x)
        x_b4_2 = F.relu(self.b4_2(x_b4_1))

        return torch.cat((x_b1, x_b2_2, x_b3_2, x_b4_2), dim=1) # Shape(B, C, H, W)
class GoogleNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )

        self.conv_2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.conv_3 = nn.Conv2d(
            in_channels=64,
            out_channels=192,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.max_pool= nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.inception_3 = nn.Sequential( # inception_3 includes two Inception modules (3a and 3b) as defined in the original GoogleNet paper.
            Inception(192, 64, (96,128), (16, 32), 32), # Shape(B, 64 + 128 + 32 + 32 = 256, H, W)
            Inception(256, 128, (128,192), (32, 96), 64) # Shape(B, 128 + 192 + 96 + 64 = 480, H, W)
        )

        self.inception_4 = nn.Sequential( # inception_4 includes five Inception modules (4a to 4e).
            Inception(480, 192, (96, 208), (16, 48), 64), # Shape(B, 192 + 208 + 48 + 64 = 512, H, W)
            Inception(512, 160, (112, 224), (24, 64), 64), # Shape(B, 160 + 224 + 64 + 64 = 512, H, W)
            Inception(512, 128, (128, 256), (24, 64), 64), # Shape(B, 128 + 256 + 64 + 64 = 512, H, W)
            Inception(512, 112, (144, 288), (32, 64), 64), # Shape(B, 112 + 288 + 32 + 64 = 528, H, W)
            Inception(528, 256, (160, 320), (32, 128), 128) # Shape(B, 256 + 320 + 128 + 128 = 832, H, W)
        )

        self.inception_5 = nn.Sequential( # inception_5 includes two Inception modules (5a and 5b).
            Inception(832, 256, (160, 320), (32, 128), 128), # Shape(B, 256 + 320 + 128 + 128 = 832, H, W)
            Inception(832, 384, (192, 384), (48, 128), 128) # Shape(B, 384 + 384 + 128 + 128 = 1024, H, W)
        )

        self.avg_pool = nn.AvgPool2d( # or nn.AdaptiveAvgPool2d((1,1)) for flexible image size, Global Average Pooling
            kernel_size=7,
            stride=1,
            padding=0
        )

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.4) # Dropout a whole channel with probability 0.4

        self.output = nn.Linear( # or nn.LazyLinear(out_features=num_classes)
            in_features=1024,
            out_features=num_classes
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv_1(x)) 
        # (bs, 64, 112, 112)
        x = self.max_pool(x) 
        # (bs, 64, 56, 56)
        x = F.relu(self.conv_2(x)) 
        # (bs, 64, 56, 56)
        x = F.relu(self.conv_3(x)) 
        # (bs, 192, 56, 56)
        x = self.max_pool(x) 
        # (bs, 192, 28, 28)
        x = self.inception_3(x) 
        # (bs, 480, 28, 28)
        x = self.max_pool(x) 
        # (bs, 480, 14, 14)
        x = self.inception_4(x) 
        # (bs, 832, 14, 14)
        x = self.max_pool(x) 
        # (bs, 832, 7, 7)
        x = self.inception_5(x) 
        # (bs, 1024, 7, 7)
        x = self.avg_pool(x) 
        # (bs, 1024, 1, 1)
        x = self.flatten(x)
        # Flatten the tensor to shape (bs, 1024)
        x = self.dropout(x) 
        # Apply Dropout (bs, 1024)
        x = self.output(x) 
        # (bs, num_classes)
        return x
    