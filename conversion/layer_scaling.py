import torch
import torch.nn as nn


class LayerScaler(nn.Module):
    def __init__(self, factor: float = 0.8):
        super().__init__()
        self.scaling_factor = factor

    def forward(self, x):
        return x

    def scale_range(self, bound: torch.Tensor, original_output: torch.Tensor):
        result = self.scaling_factor * bound
        result += (1 - self.scaling_factor) * original_output
        return result

def LinearToScaler(layer: nn.Linear):

