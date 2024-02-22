import torch
import torch.nn as nn
import copy
from typing import Tuple


def calculate_input_bounds(t: torch.Tensor, e: float) -> Tuple[torch.Tensor, torch.Tensor]:
    lb = t - e
    ub = t + e
    return lb, ub


def extract_params(layer: nn.Module):
    weights = layer.state_dict()["weight"]
    if "bias" in layer.state_dict():
        bias = layer.state_dict()["bias"]
    else:
        bias = None
    return copy.deepcopy(weights), copy.deepcopy(bias)


def split_layer(layer: nn.Module):
    weights, bias = extract_params(layer)
    pos_weights = copy.deepcopy(weights).apply_(lambda x: max(x, 0))
    neg_weights = copy.deepcopy(weights).apply_(lambda x: min(x, 0))
    neg_layer = copy.deepcopy(layer)
    pos_layer = copy.deepcopy(layer)
    if bias is not None:
        pos_layer.load_state_dict({"weight": pos_weights, "bias": bias})
        neg_layer.load_state_dict({"weight": neg_weights, "bias": bias})
    else:
        pos_layer.load_state_dict({"weight": pos_weights})
        neg_layer.load_state_dict({"weight": neg_weights})
    return neg_layer, pos_layer
