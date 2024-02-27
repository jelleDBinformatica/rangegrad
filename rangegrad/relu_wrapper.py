import torch
import torch.nn as nn

from rangegrad.base_wrapper import BaseWrapper

from typing import Union, Tuple


class ReluWrapper(BaseWrapper):
    def __init__(self, factor: float = 0.9):
        super(ReluWrapper, self).__init__()
        self.original_module = nn.ReLU()
        self.original_module2 = nn.ReLU()
        self.factor = factor

    def _scale_bounds(self, prev_y: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        new_lb = (self.factor * lb) + ((1-self.factor) * prev_y)
        new_ub = (self.factor * ub) + ((1-self.factor) * prev_y)
        return new_lb, new_ub

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor]]):
        assert self.rangegrad_mode in ["forward", "lower", "upper", "bounds"], f"invalid rangegrad mode for relu: {self.rangegrad_mode}"
        if self.rangegrad_mode == "forward":
            return self.original_module(x)
        assert type(x) != torch.Tensor, "single tensor given as input for bound propagation in ReLU"
        lb, prev_y, ub = x
        lb, ub = self._scale_bounds(prev_y, lb, ub)

        bounds = (
            self.original_module(lb),
            self.original_module(prev_y),
            self.original_module2(ub)
        )
        return bounds
