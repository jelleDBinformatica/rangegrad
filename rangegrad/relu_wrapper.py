import torch
import torch.nn as nn
import torch.nn.functional as F

from rangegrad.base_wrapper import BaseWrapper

from typing import Union, Tuple, Callable

def relu_scale_factor(lb: torch.Tensor, center: torch.Tensor, ub: torch.Tensor,
                      factor: float) -> float:
    # Original code by Sam Pinxteren
    # https://github.com/SamPinxteren/RangeGrad/blob/master/minmax/mm.py
    scale = 1
    with torch.no_grad():
        # get number of allowed elements
        # dim yi
        allowed = center.numel() * factor

        # get bound distances
        L = center / (center - lb)
        U = center / (center - ub)
        # not sure about this, maybe U > 1?
        distances = F.relu(L * (L < 1)) + F.relu(U * (U < 1))

        overcrossed = torch.sum(distances > 0).item() - allowed
        overcrossed = int(overcrossed)

        if overcrossed > 0:
            scale, index = torch.kthvalue(-distances.flatten(), overcrossed)
            scale = -scale.item()
    return scale


class ReluWrapper(BaseWrapper):
    def __init__(self,
                 scaling_func: Callable = relu_scale_factor,
                 factor: float = 0.05,):
        super(ReluWrapper, self).__init__()
        self.original_module = nn.ReLU()
        self.original_module2 = nn.ReLU()
        self.factor = factor
        self.scaling_func = scaling_func

    def _scale_bounds(self, prev_y: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        specific_factor = self.scaling_func(lb, prev_y, ub, self.factor)
        new_lb = (specific_factor * lb) + ((1-specific_factor) * prev_y)
        new_ub = (specific_factor * ub) + ((1-specific_factor) * prev_y)
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
