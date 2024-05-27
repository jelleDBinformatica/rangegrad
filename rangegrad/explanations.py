import torch
import copy

from typing import Optional

from rangegrad.module_wrapper import ModuleWrapper
from rangegrad.vgg_translation import TranslatedVGG
from utils.various import adaptive_cuda


def rangegrad_explanation(
        model: TranslatedVGG,
        x: torch.Tensor,
        bound_range: float,
        scaling_factor: Optional[float] = None
):
    if scaling_factor is not None:
        model.set_scaling_factor(factor=scaling_factor)
    output_bounds = []
    for bound_index in range(2):
        x = torch.autograd.Variable(x, requires_grad=True)
        x.requires_grad = True
        if torch.cuda.is_available():
            x = adaptive_cuda(x)
        x.retain_grad()

        # first, get supposed output of model
        model.set_to_forward()
        y1 = model(x)

        model.set_to_bounds()

        with torch.no_grad():
            lb = x - bound_range
            ub = x + bound_range
        relevant_input = [lb, ub][bound_index]
        bounds = model((lb, x, ub))

        relevant_bound = torch.flatten(bounds[2*bound_index])
        relevant_bound.retain_grad()

        relevant_bound.backward(torch.ones_like(relevant_bound))
        output_bounds.append(copy.deepcopy(x.grad).detach())
    grad_diff = output_bounds[1] - output_bounds[0]
    grad_diff = torch.sum(grad_diff, 1)
    return grad_diff.detach()