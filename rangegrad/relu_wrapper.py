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

    def _scale_bounds(self, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tensor_distance: torch.Tensor = ub - lb
        tensor_distance *= self.factor
        tensor_center: torch.Tensor = (ub + lb) / 2.0
        new_lb = tensor_center - tensor_distance
        new_ub = tensor_center + tensor_distance
        return new_lb, new_ub

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor]]):
        assert self.rangegrad_mode in ["forward", "lower", "upper", "boundse"], f"invalid rangegrad mode for relu: {self.rangegrad_mode}"
        if self.rangegrad_mode == "forward":
            return self.original_module(x)
        assert type(x) != torch.Tensor, "single tensor given as input for bound propagation in ReLU"
        lb, ub = x

        bounds = self.original_module(lb), self.original_module(ub)
        # xlb, xub = bounds
        # bounds = self._scale_bounds(xlb, xub)
        return bounds
