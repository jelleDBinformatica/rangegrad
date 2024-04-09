import torch
import torch.nn as nn
import torch.functional as F

from typing import List

from rangegrad.base_wrapper import BaseWrapper
from rangegrad.linear_wrapper import LinearWrapper


def combine_linear_modules(l1: nn.Module,
                           l2: nn.Module):
    in_features = l1.weight.shape[-1]
    out_features = l2.weight.shape[-2]

    comp_weight = torch.mm(l2.weight, l1.weight)

    bias2 = l2.bias if l2.bias is not None else 0
    comp_bias = l1.bias
    comp_bias = bias2 + torch.matmul(l2.weight, comp_bias)

    new_linear = nn.Linear(in_features, out_features)
    new_linear.weight.data = comp_weight
    new_linear.bias.data = comp_bias
    return new_linear




class Composition(nn.Module):
    def __init__(self):
        super(Composition).__init__()
        self.intermediary_modules: nn.ModuleList = nn.ModuleList()
        self.composition = nn.Sequential()
        
    def _add_module(self, new_module: nn.Module):
        current_tail_module = None
        if len(self.intermediary_modules) > 0:
            current_tail_module = self.intermediary_modules[-1]

        can_merge = type(current_tail_module) in [nn.Linear, LinearWrapper]
        can_merge = can_merge and type(new_module) == nn.Linear
        if can_merge:
