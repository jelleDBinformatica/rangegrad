import torch
import torch.nn as nn


class BaseWrapper(nn.Module):
    def __init__(self):
        super(BaseWrapper, self).__init__()
        # "forward", "upper", "lower"
        self.rangegrad_mode: str = "forward"

    def upper_bound(self, lower_input, upper_input):
        raise NotImplementedError

    def lower_bound(self, lower_input, upper_input):
        raise NotImplementedError

    def set_to_forward(self):
        self.rangegrad_mode = "forward"

    def set_to_upper(self):
        self.rangegrad_mode = "upper"

    def set_to_lower(self):
        self.rangegrad_mode = "lower"

    def set_to_bounds(self):
        self.rangegrad_mode = "bounds"
