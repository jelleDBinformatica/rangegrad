import torch
import torch.nn as nn

from rangegrad.module_wrapper import BaseWrapper
from conversion.utils import split_layer

from typing import Union, Tuple


class LinearWrapper(BaseWrapper):
    def __init__(self, original_module: nn.Module):
        super(LinearWrapper, self).__init__()
        if type(original_module) not in [nn.Linear, nn.Conv2d]:
            raise TypeError(f"original layer type {type(original_module)} not suitable for wrapper.")
        self.original_module = original_module
        neg_layer, pos_layer = split_layer(original_module)
        self.neg_layer = neg_layer
        self.pos_layer = pos_layer

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor]]):
        if self.rangegrad_mode == "forward":
            y = self.original_module(x)
            return y
        try:
            lb, prev_y, ub = x
            y = (self.lower_bound(lb, ub), self.original_module(prev_y), self.upper_bound(lb, ub))
        except Exception:
            raise Exception("error occurred in linear bound propagation")
        return y

    def lower_bound(self, lower_input: torch.Tensor, upper_input: torch.Tensor):
        olb = self.neg_layer(upper_input) + self.pos_layer(lower_input)
        # olb -= self.original_module.bias
        return olb

    def upper_bound(self, lower_input: torch.Tensor, upper_input: torch.Tensor):
        oub = self.neg_layer(lower_input) + self.pos_layer(upper_input)
        # oub -= self.original_module.bias
        return oub
