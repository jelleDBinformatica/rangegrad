import torch
import torch.nn as nn
import copy

from rangegrad.translation import translate_any_model
from rangegrad.module_wrapper import ModuleWrapper
from rangegrad.explanations import rangegrad_explanation
from utils.various import adaptive_cuda


class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.lin1 = nn.Linear(2, 2, bias=True)
        bias = torch.Tensor([1, 1])
        self.lin2 = nn.Linear(2, 1, bias=False)
        self.lin1.load_state_dict({"weight": torch.Tensor([[1, 2], [-3, 4]]),
                                   "bias": bias})
        self.lin2.load_state_dict({"weight": torch.Tensor([[1, 2]])})
        self.seq = nn.Sequential(
            self.lin1,
            nn.ReLU(),
            self.lin2
        )

    def forward(self, x):
        return self.seq(x)


x = torch.Tensor([[2, 1]])
x = torch.autograd.Variable(x)
x.requires_grad = True
x.retain_grad()

e = 1
xlb = x - e
xub = x + e

outputs_per_model = {
    Model1: [torch.Tensor([]), torch.Tensor([])]
}

if __name__ == "__main__":
    for constr, outputs in outputs_per_model.items():
        model: ModuleWrapper = translate_any_model(constr())
        # model.set_debug(True)

        y = model(x)
        print(y)
        model.set_to_explin()
        lb, _, ub = model.forward((xlb, x, xub))
        lb.retain_grad()
        ub.retain_grad()

        lb.backward(torch.ones_like(lb))
        print(lb, x.grad)

        lbg = copy.deepcopy(x.grad.detach())

        lb, _, ub = model.forward((xlb, x, xub))
        ub[0].backward()
        print(ub, x.grad)
        ubg = copy.deepcopy(x.grad.detach())

        print(ubg, lbg)
        print(ubg - lbg)

        model = adaptive_cuda(model)
        bleh = rangegrad_explanation(model, x, e)
