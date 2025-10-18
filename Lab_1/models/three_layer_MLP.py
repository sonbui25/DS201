import torch
from torch import nn
class ThreeLayerMLP(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        # Initialize model layers
        self.block_1 = nn.Sequential(
            nn.Linear(in_features=input_shape, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=output_shape)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define forward process
        x = x.view(x.shape[0], -1)  # shape [32, 28, 28] -> [32, 784] (batch_size, input_shape)
        return self.block_1(x)