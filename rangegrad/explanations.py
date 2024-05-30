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
        scaling_factor: Optional[float] = None,
        target: Optional[int] = None,
        explin_override: bool = False
):
    if scaling_factor is not None:
        model.set_scaling_factor(factor=scaling_factor)
    x = torch.autograd.Variable(x)
    x.requires_grad = True
    x.retain_grad()
    x = adaptive_cuda(x)

    model.zero_grad()
    if target is None:
        with torch.no_grad():
            model.set_to_forward()
            x1 = model(x)
            target = torch.argmax(x1).item()

    OH = adaptive_cuda(torch.zeros((1, 1000)))
    OH[0, target] = 1

    model.zero_grad()

    if explin_override:
        model.set_to_explin()
    else:
        model.set_to_bounds()

    diff_matrix = torch.full(x.shape, float(bound_range), requires_grad=True)
    diff_matrix = adaptive_cuda(diff_matrix)
    diff_matrix.retain_grad()

    lb = x - diff_matrix
    ub = x + diff_matrix
    print(lb, ub, diff_matrix)


    bounds = model((lb, x, ub))

    f = (bounds[2] - bounds[0])

    f.backward(OH)
    grad_diff = diff_matrix.grad.data.squeeze().detach()

    return grad_diff