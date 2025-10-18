import torch
from torch import nn
class OneLayerMLP(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        # Initialize model layers
        self.layer_1 = nn.Linear(in_features=input_shape, out_features=output_shape)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define forward process
        x = x.view(x.shape[0], -1)  # shape [32, 28, 28] -> [32, 784] (batch_size, input_shape)
        return self.layer_1(x)