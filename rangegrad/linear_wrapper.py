import torch
import torch.nn as nn
import torch.nn.functional as F

from rangegrad.module_wrapper import BaseWrapper
from conversion.utils import split_layer
from utils.various import adaptive_cuda

from typing import Union, Tuple


class LinearWrapper(BaseWrapper):
    def __init__(self, original_module: nn.Module):
        super(LinearWrapper, self).__init__()
        if type(original_module) not in [nn.Linear, nn.Conv2d]:
            raise TypeError(f"original layer type {type(original_module)} not suitable for wrapper.")
        self.original_module = original_module
        # neg_layer, pos_layer = split_layer(original_module)
        # self.neg_layer = neg_layer
        # self.pos_layer = pos_layer
        self.bias = None

        with torch.no_grad():
            self.neg_weights = adaptive_cuda(-F.relu(-original_module.weight.data))
            self.pos_weights = adaptive_cuda(F.relu(original_module.weight.data))
            if original_module.bias is not None:
                self.bias = adaptive_cuda(original_module.bias.clone())

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor]]):
        if self.rangegrad_mode == "forward":
            y = self.original_module(x)
            return y
        try:
            lb, prev_y, ub = x
            y = self.bounds(lb, prev_y, ub)
            self.debug_print(f"difference: "
                             f"{torch.sum(torch.lt((y[2]-y[0]), 0))}")
        except Exception as e:
            print(e)
            raise Exception("error occurred in linear bound propagation")
        return y

    def bounds(self, lb: torch.Tensor, x: torch.Tensor, ub: torch.Tensor):
        if isinstance(self.original_module, nn.Linear):
            y = self.original_module(x)
            return self.lower_bound(lb, ub), y, self.upper_bound(lb, ub)
        elif isinstance(self.original_module, nn.Conv2d):
            return self.conv_bounds(lb, x, ub)
        else:
            print("time to panic, instance found:", type(self.original_module))
            exit()

    def lower_bound(self, lower_input: torch.Tensor, upper_input: torch.Tensor):
        # olb = self.neg_layer(upper_input) + self.pos_layer(lower_input)
        # if self.original_module.bias is not None:
        #     self.debug_print(f'adding bias of {self.original_module.bias} to lb')
        #     olb -= self.original_module.bias
        olb = F.linear(lower_input, self.pos_weights, self.bias) + F.linear(upper_input, self.neg_weights)
        f = self.bias if self.bias is not None else 0
        self.debug_print(f'sanity_check: '
                         f'{olb-(F.linear(lower_input, self.pos_weights) + F.linear(upper_input, self.neg_weights)) + f}')
        return olb

    def upper_bound(self, lower_input: torch.Tensor, upper_input: torch.Tensor):
        # oub = self.neg_layer(lower_input) + self.pos_layer(upper_input)
        #
        # if self.original_module.bias is not None:
        #     self.debug_print(f'adding bias of {self.original_module.bias} to ub')
        #     oub -= self.original_module.bias
        oub = F.linear(upper_input, self.pos_weights, self.bias) + F.linear(lower_input, self.neg_weights)
        return oub

    def conv_bounds(self, lower_input: torch.Tensor, x: torch.Tensor, upper_input: torch.Tensor):
        # positive weights
        with torch.no_grad():
            W_p = F.relu(self.original_module.weight)
            # OPPOSITE of negative weights
            W_n = F.relu(-self.original_module.weight)

        # conv_module
        cm = self.original_module
        center = cm(x)

        # lb * positive weights
        min_p = cm._conv_forward(lower_input, W_p, cm.bias)
        # lb * (- negative_weights) -> invert sign
        min_n = cm._conv_forward(lower_input, W_n, None)
        # ub * positive weights
        max_p = cm._conv_forward(upper_input, W_p, cm.bias)
        # ub * (- negative_weights) -> invert sign
        max_n = cm._conv_forward(upper_input, W_n, None)

        out = min_p - max_n, center, max_p - min_n
        return out