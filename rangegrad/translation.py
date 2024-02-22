import torch.nn as nn
import torch

from rangegrad.base_wrapper import BaseWrapper
from rangegrad.linear_wrapper import LinearWrapper
from rangegrad.module_wrapper import ModuleWrapper
from rangegrad.relu_wrapper import ReluWrapper
from rangegrad.custom_model import CustomModel


def translate_module(module: nn.Module):
    new_module = module
    if type(module) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        new_module = LinearWrapper(module)
    if type(module) == nn.Sequential:
        new_module = translate_any_model(module)
        new_module = ModuleWrapper(new_module)
    if type(module) == nn.ReLU:
        new_module = ReluWrapper()
    # TODO: put everything into ModuleWrapper
    else:
        new_module = ModuleWrapper(module)
    return new_module


def translate_any_model(model: nn.Module):
    for name, module in model.named_children():
        if module == model:
            continue
        if isinstance(module, BaseWrapper):
            continue
        new_module = translate_module(module)
        model.__setattr__(name, new_module)
    return ModuleWrapper(model)


if __name__ == "__main__":
    # inner_seq = nn.Sequential(
    #     nn.Linear(2, 5),
    #     nn.ReLU(),
    #     nn.Linear(5, 1)
    # )
    # outer_seq = nn.Sequential(
    #     nn.Linear(2, 6),
    #     nn.ReLU(),
    #     nn.Linear(6, 2),
    #     inner_seq
    # )
    # model = nn.Sequential(
    #     nn.Linear(4, 1),
    #     nn.ReLU(),
    #     nn.Linear(1, 2),
    #     nn.ReLU(),
    #     outer_seq
    # )
    model = CustomModel()
    model: BaseWrapper = translate_any_model(model)

    x = torch.Tensor([1, 1, 1, 1])
    # y = model(x)
    # print(y)

    ub = torch.Tensor([1, 1, 1, 1]) + 1
    lb = torch.Tensor([1, 1, 1, 1]) - 1
    model.set_to_upper()
    u, l = model((lb, ub))

    print(u)
    print(l)
