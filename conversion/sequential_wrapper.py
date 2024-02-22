import torch
import torch.nn as nn

from typing import List, Union
import copy

from conversion.layer_wrapper import BaseWrapper
from conversion.linear_wrapper import LinearWrapper
from conversion.utils import split_layer


def translate_module(name: str, module: nn.Module):
    if type(module) in [nn.Linear, nn.Conv2d]:
        new_module = LinearWrapper(module)
    if type(module) == nn.Sequential:
        new_module = translate_any_model(module)
        new_module = SequentialWrapper(new_module)
    return new_module


def translate_any_model(model: nn.Module):
    for name, module in model.named_modules():
        if module == model:
            continue
        new_module = translate_module(name, module)
        model.__setattr__(name, new_module)
    return model


class SequentialWrapper(nn.Module, BaseWrapper):
    def __init__(self, original_module: nn.Sequential):
        super(SequentialWrapper, self).__init__()
        self.original_module = original_module
        self.split_modules: List[Union[nn.Module, BaseWrapper]] = []
        self.translate()

    def translate(self):
        translated = []
        self.split_modules = []
        for module in self.original_module.modules():
            if module in translated or module == self.original_module:
                continue
            translated.append(module)
            if type(module) in [nn.Linear, nn.Conv2d]:
                self.split_modules.append(LinearWrapper(module))
            elif type(module) == nn.Sequential:
                for submodule in module.modules():
                    translated.append(submodule)
                self.split_modules.append(SequentialWrapper(module))
            else:
                self.split_modules.append(module)

    def forward(self, x):
        return self.original_module(x)

    def bounds(self, lower_input, upper_input):
        output_lb = None
        output_ub = None
        output_lb = lower_input
        output_ub = upper_input
        for i, submodule in enumerate(self.split_modules):
            if not isinstance(submodule, BaseWrapper):
                print(type(submodule))
                intermediary_lb = submodule(output_lb)
                intermediary_ub = submodule(output_ub)
            else:
                submodule: BaseWrapper = submodule
                intermediary_lb = submodule.lower_bound(output_lb, output_ub)
                intermediary_ub = submodule.upper_bound(output_lb, output_ub)
            output_ub = intermediary_ub
            output_lb = intermediary_lb
            output_ub.retain_grad()
            output_lb.retain_grad()

            print(output_lb, output_ub)

        return output_lb, output_ub

    def lower_bound(self, lower_input, upper_input):
        return self.bounds(lower_input, upper_input)[0]

    def upper_bound(self, lower_input, upper_input):
        return self.bounds(lower_input, upper_input)[1]


if __name__ == "__main__":
    temp = nn.Sequential(
        nn.Linear(2, 1),
        nn.ReLU(),
        nn.Linear(1, 1)
    )
    for param in temp.parameters():
        print(param)
    print("-------------------")

    input = torch.Tensor([[0, 0]])
    input.requires_grad = True
    input.retain_grad()

    wrapper = SequentialWrapper(temp)
    # for module in wrapper.split_modules:
    #     print(module)

    lb, ub = wrapper.bounds(input-1, input+1)
    lb.retain_grad()
    ub.retain_grad()

    print(f"lower bound: {lb}")
    lb.backward()
    print(input.grad)

    print(f"upper bound: {ub}")
    ub.backward()
    print(input.grad)
