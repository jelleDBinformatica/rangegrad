import torch
import torch.nn as nn

from typing import List

from conversion.linear_wrapper import LinearWrapper


def construct_conv():
    weights = torch.Tensor([[[[1, 1], [-1, -1]]]])
    bias = torch.Tensor([1])
    result = nn.Conv2d(1, 1, 2)
    result.load_state_dict({
        "weight": weights,
        "bias": bias
    })
    return result


class ModelTranslator:
    def __init__(self):
        self.translated: List[nn.Module] = []

    def translate(self, module: nn.Module):
        if module in self.translated:
            return None
        if type(module) in [nn.Linear, nn.Conv2d]:
            return LinearWrapper(module)


if __name__ == "__main__":
    translator = ModelTranslator()
    f = construct_conv()
    x = torch.Tensor([[[1, 2, 3],
                       [1, 2, 3]]])

    w = translator.translate(f)
    print(f(x))
    print(w(x))