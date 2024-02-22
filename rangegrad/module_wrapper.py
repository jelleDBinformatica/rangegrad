import torch
import torch.nn as nn

from rangegrad.base_wrapper import BaseWrapper


class ModuleWrapper(BaseWrapper):
    def __init__(self, original_module: nn.Module):
        super(ModuleWrapper, self).__init__()
        self.original_module = original_module

    def forward(self, x):
        if self.rangegrad_mode == "forward":
            return self.original_module(x)
        try:
            y = self.original_module(x)
            return y
        except Exception:
            lb, ub = x
            y = (self.original_module(lb), self.original_module(ub))
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
