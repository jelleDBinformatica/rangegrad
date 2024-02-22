import torch
import torch.nn as nn

from conversion.layer_wrapper import BaseWrapper
from conversion.utils import split_layer


class LinearWrapper(BaseWrapper, nn.Module):
    def __init__(self, original_module: nn.Module):
        super().__init__()
        if type(original_module) not in [nn.Linear, nn.Conv2d]:
            raise TypeError(f"original layer type {type(original_module)} not suitable for wrapper.")
        self.original_module = original_module
        neg_layer, pos_layer = split_layer(original_module)
        self.neg_layer = neg_layer
        self.pos_layer = pos_layer

    def forward(self, x: torch.Tensor):
        return self.original_module(x)

    def lower_bound(self, lower_input: torch.Tensor, upper_input: torch.Tensor):
        olb = self.neg_layer(upper_input) + self.pos_layer(lower_input)
        # olb -= self.original_module.bias
        return olb

    def upper_bound(self, lower_input: torch.Tensor, upper_input: torch.Tensor):
        oub = self.neg_layer(lower_input) + self.pos_layer(upper_input)
        # oub -= self.original_module.bias
        return oub
