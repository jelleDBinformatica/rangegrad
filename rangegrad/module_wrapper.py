import torch
import torch.nn as nn

from rangegrad.base_wrapper import BaseWrapper

from typing import Callable


class ModuleWrapper(BaseWrapper):
    def __init__(self, original_module: nn.Module):
        super(ModuleWrapper, self).__init__()
        self.original_module = original_module

    def forward(self, x):
        if self.rangegrad_mode == "forward":
            return self.original_module(x)
        try:
            y = self.original_module(x)
        except Exception:
            lb, prev_y, ub = x
            y = (
                self.original_module(lb),
                self.original_module(prev_y),
                self.original_module(ub)
            )
        return y

    def upper_bound(self, lower_input, upper_input):
        if isinstance(self.original_module, BaseWrapper):
            return self.original_module.upper_bound(lower_input, upper_input)
        else:
            return self.original_module(upper_input)

    def lower_bound(self, lower_input, upper_input):
        if isinstance(self.original_module, BaseWrapper):
            return self.original_module.lower_bound(lower_input, upper_input)
        else:
            return self.original_module(lower_input)

    def __call__(self, x):
        return self.forward(x)

    def set_to_forward(self):
        super().set_to_forward()
        for module in self.modules():
            if module == self:
                continue
            if isinstance(module, BaseWrapper):
                module.set_to_forward()

    def set_to_upper(self):
        super().set_to_upper()
        for module in self.modules():
            if module == self:
                continue
            if isinstance(module, BaseWrapper):
                module.set_to_upper()

    def set_to_lower(self):
        super().set_to_lower()
        for module in self.modules():
            if module == self:
                continue
            if isinstance(module, BaseWrapper):
                module.set_to_lower()

    def set_to_bounds(self):
        super().set_to_bounds()
        for module in self.modules():
            if module == self:
                continue
            if isinstance(module, BaseWrapper):
                module.set_to_bounds()

    def set_to_explin(self):
        super().set_to_explin()
        for module in self.modules():
            if module == self:
                continue
            if isinstance(module, BaseWrapper):
                module.set_to_explin()

    def set_to_init(self):
        super().set_to_init()
        for module in self.modules():
            if module == self:
                continue
            if isinstance(module, BaseWrapper):
                module.set_to_init()

    def set_debug(self, mode: bool):
        super().set_debug(mode)
        for module in self.modules():
            if module == self:
                continue
            if isinstance(module, BaseWrapper):
                module.set_debug(mode)

    def set_scaling_factor(self, factor: float):
        for module in self.modules():
            if module == self:
                continue
            if isinstance(module, BaseWrapper):
                module.set_scaling_factor(factor)

    def get_device(self):
        return self.original_module.get_device()
