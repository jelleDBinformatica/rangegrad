import torch
import torch.nn as nn
import torch.nn.functional as F

from rangegrad.base_wrapper import BaseWrapper

from typing import Union, Tuple, Callable, Optional


def relu_scale_factor(self, lb: torch.Tensor, center: torch.Tensor, ub: torch.Tensor,
                      factor: float) -> float:
    # Original code by Sam Pinxteren
    # https://github.com/SamPinxteren/RangeGrad/blob/master/minmax/mm.py
    scale = 1
    with torch.no_grad():
        # get number of allowed elements
        # dim yi
        allowed = center.numel() * factor
        self.debug_print((center.numel(), factor))
        self.debug_print(("allowed", allowed))

        # get bound distances
        L = center / (center - lb)
        U = center / (center - ub)
        # not sure about this, maybe U > 1?
        distances = F.relu(L * (L < 1)) + F.relu(U * (U < 1))

        overcrossed = torch.sum(distances > 0).item() - allowed
        self.debug_print(overcrossed)
        overcrossed = int(overcrossed)

        if overcrossed > 0:
            scale, index = torch.kthvalue(-distances.flatten(), overcrossed)
            scale = -scale.item()
            self.debug_print(f" post-relu scaling happened with factor {scale}")
    return scale


class Rangegrad_ReluWrapper(BaseWrapper):
    def __init__(self,
                 scaling_func: Callable = relu_scale_factor,
                 factor: float = 0.05):
        super(Rangegrad_ReluWrapper, self).__init__()
        self.original_module = nn.ReLU()
        self.lower_module = nn.ReLU()
        self.upper_module = nn.ReLU()
        self.factor = factor
        self.scaling_func = scaling_func

    def _scale_bounds(self, prev_y: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        specific_factor = self.scaling_func(self, lb, prev_y, ub, self.factor)
        new_lb = (specific_factor * lb) + ((1 - specific_factor) * prev_y)
        new_ub = (specific_factor * ub) + ((1 - specific_factor) * prev_y)
        return new_lb, new_ub

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor]]):
        assert self.rangegrad_mode in ["forward",
                                       "bounds"], f"invalid rangegrad mode for relu: {self.rangegrad_mode}"
        if self.rangegrad_mode == "forward":
            return self.original_module(x)
        assert type(x) != torch.Tensor, "single tensor given as input for bound propagation in ReLU"
        lb, prev_y, ub = x
        lb, ub = self._scale_bounds(prev_y, lb, ub)

        bounds = (
            self.lower_module(lb),
            self.original_module(prev_y),
            self.upper_module(ub)
        )
        return bounds

    def set_scaling_factor(self, factor: float):
        self.factor = factor


def get_bound_argmax(mask: Tuple[torch.Tensor, torch.Tensor], composition_matrix):
    summed_layer = torch.sum(composition_matrix, 0)

    gt = torch.gt(summed_layer, 0).int()
    gt = gt.unsqueeze(0)

    bound_tensor = torch.concat((mask[0], mask[1]))
    choice_tensor = torch.cat((1 - gt, gt), 0)

    xiM = torch.sum(bound_tensor * choice_tensor, 0)
    # xim = torch.sum(bound_tensor * (1 - choice_tensor), 0)
    return xiM


def get_bound_argmin(mask: Tuple[torch.Tensor, torch.Tensor], composition_matrix):
    summed_layer = torch.sum(composition_matrix, 0)

    gt = torch.gt(summed_layer, 0).int()
    gt = gt.unsqueeze(0)

    bound_tensor = torch.concat((mask[0], mask[1]))
    choice_tensor = torch.cat((1 - gt, gt), 0)

    xim = torch.sum(bound_tensor * (1 - choice_tensor), 0)
    return xim


class ReluWrapper(Rangegrad_ReluWrapper):
    def __init__(self, scaling_func: Callable = relu_scale_factor, factor: float = 0.05):
        super().__init__(scaling_func, factor)

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor]],
                previous_module: Optional[BaseWrapper] = None):
        assert self.rangegrad_mode in [
            "forward", "bounds", "explin"
        ], f"Explin ReLUWrapper got invalid mode {self.rangegrad_mode}"

        if self.rangegrad_mode != "explin":
            return super().forward(x)
        # now we can assume explin propagation, which also requires the previous module
        lb, x_, ub = x
        lb, ub = self._scale_bounds(x_, lb, ub)
        # xiM = ub
        # xim = lb

        # Mi = torch.max(ub, dim=-1).values.unsqueeze(-1)
        #
        # mi = torch.min(lb, dim=-1).values.unsqueeze(-1)
        #
        #
        # yub = ub - mi
        # frac = F.relu(Mi) - F.relu(mi)
        # frac /= (Mi - mi)
        # x_u = frac * yub + F.relu(mi)
        # x_l = F.relu(lb)

        # print(x_l, x_u)
        with torch.no_grad():
            slope = torch.zeros(x_.shape) + ub
            slope_denom = ub - lb
            slope = slope / slope_denom
            u_slope = torch.nan_to_num(slope, 0, 1, 0)

            bias_enabler = torch.gt(ub, 0) * torch.le(lb, 0)
            u_bias = - u_slope * lb * bias_enabler
        x_u = u_slope * ub + u_bias
        x_l = F.relu(lb)

        # print(F.linear(x_u, u_slope, bias=bias))
        # slope = torch.zeros(size=x.shape[0])

        return (
            x_l,
            F.relu(x_),
            x_u
        )


