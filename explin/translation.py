import torch
import torch.nn as nn
import torch.functional as F


def translate_weights_to_lin_relu(
        lin_weights: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor
):
    """
    given linear weights, compose a set of weights for
    :param lin_weights:
    :param lb:
    :param ub:
    :return:
    """
    ...
