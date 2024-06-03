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
        self.lin1 = adaptive_cuda(nn.Linear(2, 2, bias=True))
        bias = (torch.Tensor([1, 1]))
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
x = adaptive_cuda(x)
x.requires_grad = True
x.retain_grad()

val = 1.
e = adaptive_cuda(torch.full(x.shape, val, requires_grad=True))
e.retain_grad()
xlb = adaptive_cuda(x - e)
xub = adaptive_cuda(x + e)

outputs_per_model = {
    Model1: [torch.Tensor([]), torch.Tensor([])]
}

if __name__ == "__main__":
    for constr, outputs in outputs_per_model.items():
        model: ModuleWrapper = adaptive_cuda(translate_any_model(constr()))
        model.set_debug(False)

        y = model(x)
        print(y)
        model.set_to_explin()
        lb, _, ub = model.forward((xlb, x, xub))
        lb.retain_grad()
        ub.retain_grad()

        lb.backward(adaptive_cuda(torch.ones_like(lb)))
        print(lb, x.grad)

        lbg = copy.deepcopy(e.grad.detach())

        lb, _, ub = model.forward((xlb, x, xub))
        ub.backward(adaptive_cuda(torch.ones_like(lb)))
        print(ub, e.grad)
        ubg = copy.deepcopy(e.grad.detach())

        print(ubg, lbg)
        print(ubg - lbg)

        model = adaptive_cuda(model)
        bleh = rangegrad_explanation(model, x, val, num_classes=1)
