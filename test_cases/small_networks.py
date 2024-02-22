import torch
import torch.nn as nn
import copy

from rangegrad.translation import translate_any_model
from rangegrad.module_wrapper import ModuleWrapper


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.lin1 = nn.Linear(2, 2, bias=False)
        self.lin2 = nn.Linear(2, 1, bias=False)
        self.lin1.load_state_dict({"weight": torch.Tensor([[1, 2], [-3, 4]])})
        self.lin2.load_state_dict({"weight": torch.Tensor([[1, 2]])})
        self.seq = nn.Sequential(
            self.lin1,
            nn.ReLU(),
            self.lin2
        )

    def forward(self, x):
        return self.seq(x)


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(1, 2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2, 1),
            nn.Softmax()
        )

    def forward(self, x):
        x = torch.Tensor([[[0, 0], [0, 0]]])
        return self.seq(x)


x = torch.Tensor([2, 1])
x = torch.autograd.Variable(x)
x.requires_grad = True
x.retain_grad()

e = 1
xlb = x - e
xub = x + e

outputs_per_model = {
    # Model1: [torch.Tensor([]), torch.Tensor([])],
    Model2: [torch.Tensor([]), torch.Tensor([])],
}

if __name__ == "__main__":
    for constr, outputs in outputs_per_model.items():
        model: ModuleWrapper = translate_any_model(constr())
        if constr == Model2:
            for mod in model.modules():
                print(mod)

        y = model(x)
        model.set_to_lower()
        lb, ub = model.forward((xlb, xub))
        lb.retain_grad()
        ub.retain_grad()
        # print(model.lower_bound(xlb, xub))
        # print(model.upper_bound(xlb, xub))
        print(y)
        lb[0].backward()
        print(lb, x.grad)

        lbg = copy.deepcopy(x.grad.detach())

        lb, ub = model.forward((xlb, xub))
        ub[0].backward()
        print(ub, x.grad)
        ubg = copy.deepcopy(x.grad.detach())

        print(ubg, lbg)
        print(ubg - lbg)