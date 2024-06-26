from rangegrad.translation import translate_any_model, translate_module
from rangegrad.module_wrapper import ModuleWrapper

from torchvision.models import vgg as vggn
import torch.nn as nn
import torch


class TranslatedVGG(ModuleWrapper):
    def __init__(self, original_module: vggn.VGG):
        super(TranslatedVGG, self).__init__(original_module)
        self.original_module = original_module
        self.features = translate_any_model(original_module.features)
        self.avgpool = translate_any_model(original_module.avgpool)
        self.flatten = translate_module(nn.Flatten(1))
        self.classifier = translate_any_model(original_module.classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rangegrad_mode == "forward":
            return self.original_module(x)
        for mod in [self.features, self.avgpool, self.flatten, self.classifier]:
            try:
                y = mod(x)
            except Exception:
                lb, x_, ub = x
                y = (mod(lb), mod(x_), mod(ub))
            x = y
        return y

    def set_scaling_factor(self, factor: float):
        super().set_scaling_factor(factor)



