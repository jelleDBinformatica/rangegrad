import torch
import torch.nn as nn
import torchvision.models as models
import copy
from typing import List, Optional, Tuple

from conversion.utils import calculate_input_bounds


class Temp(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 1),
            nn.ReLU()
        )
        for mod in self.modules():
            print(mod)

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    x = torch.Tensor([[0, 0, 0]])
    lb, ub = calculate_input_bounds(x, 1)
    f = Temp()
    f(x)
