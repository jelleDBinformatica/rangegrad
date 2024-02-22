from conversion.sequential_wrapper import SequentialWrapper


import torch
import torch.nn as nn

if __name__ == "__main__":
    temp = nn.Sequential(
        nn.Linear(4, 1),
        nn.ReLU(),
        nn.Linear(1, 2)
    )

    x = torch.Tensor([0.5, 0.5, 0.5, 0.5])
    x = torch.autograd.Variable(x)
    x.requires_grad = True
    x.retain_grad()

    wrapper = SequentialWrapper(temp)
    lb: torch.Tensor = wrapper.lower_bound(x-1, x+1)
    lb.retain_grad()

    print(f"lower bound: {lb}")
    print(lb.shape)
    lb[0].backward()
    print(x.grad)

    ub = wrapper.upper_bound(x-1, x+1)
    ub.retain_grad()

    print(f"upper bound: {ub}")
    ub[0].backward()
    print(x.grad)