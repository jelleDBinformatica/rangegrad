import torch
import torch.nn as nn


class BaseWrapper(nn.Module):
    def __init__(self):
        super(BaseWrapper, self).__init__()
        # "forward", "upper", "lower"
        self.rangegrad_mode: str = "forward"
        self.debug: bool = False

    def upper_bound(self, lower_input, upper_input):
        return self.forward(upper_input)

    def lower_bound(self, lower_input, upper_input):
        return self.forward(lower_input)

    def set_to_forward(self):
        self.rangegrad_mode = "forward"

    def set_to_upper(self):
        self.rangegrad_mode = "upper"

    def set_to_lower(self):
        self.rangegrad_mode = "lower"

    def set_to_bounds(self):
        self.rangegrad_mode = "bounds"

    def set_to_init(self):
        self.rangegrad_mode = "init"

    def set_debug(self, mode: bool):
        self.debug = mode

    def debug_print(self, content):
        if self.debug:
            print(content)

    def set_to_explin(self):
        self.rangegrad_mode = "explin"

    def set_scaling_factor(self, factor: float):
        ...
