import torch
from torch import nn
class LeNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv_1 = nn.Conv2d( # 32*32 -> 28*28
            in_channels=1,
            out_channels=6,
            kernel_size=(5,5), # Can set to 5 if square kernel
            stride=1,
            padding=(2,2) # Height padding, Width padding (Height = Top + Bottom, Width = Left + Right)
        )
        self.pooling_1 = nn.AvgPool2d( # 28*28 -> 14*14
            kernel_size=(2,2),
            stride=2,
            padding=0 # No padding for pooling layer
        )  
        self.conv_2 = nn.Conv2d( # 14*14 -> 10*10
            in_channels=6,
            out_channels=16,
            kernel_size=(5,5),
            stride=1,
            padding=0
        )
        self.pooling_2 = nn.AvgPool2d( # 10*10 -> 5*5
            kernel_size=(2,2),
            stride=2,
            padding=0
        )
        self.conv_3 = nn.Conv2d( # 5*5 -> 1*1 with 120 channels, which equals to a 1D vector of size 120
            in_channels=16,
            out_channels=120,
            kernel_size=(5,5),
            stride=1,
            padding=0
        )
        self.fc = nn.Linear( # Fully Connected Layer
            in_features=120,
            out_features=84
        )
        self.output = nn.Linear( # Output Layer
            in_features=84,
            out_features=num_classes
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x) # Input x shape: (batch_size, 1, 28, 28)
        x = self.pooling_1(torch.sigmoid(x)) #(bs, 6, 14, 14)
        x = self.conv_2(x) # (bs, 16, 10, 10)
        x = self.pooling_2(torch.sigmoid(x)) #(bs, 16, 5, 5)
        x = self.conv_3(x) # (bs, 120, 1, 1)
        x = x.view(x.size(0), -1) # or x.unsqueeze(-1).unsqueeze(-1) Flatten the tensor to shape (batch_size, 120)
        x = self.fc(torch.sigmoid(x))
        x = torch.sigmoid(x)
        x = self.output(x)
        return x