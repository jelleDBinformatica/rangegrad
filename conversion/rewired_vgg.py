from torchvision.models import vgg as vggn

from conversion.sequential_wrapper import SequentialWrapper
from conversion.linear_wrapper import LinearWrapper

import torch


def rewire_vgg(original_model: vggn.VGG):
    new_features = SequentialWrapper(original_model.features)
    new_classifier = SequentialWrapper(original_model.classifier)

    original_model.features = new_features
    original_model.classifier = new_classifier


def lb_vgg(model: vggn.VGG, lb: torch.Tensor, ub: torch.Tensor):
    olb, oub = model.features.bounds(lb, ub)
    olb = model.avgpool(olb)
    oub = model.avgpool(oub)
    olb = torch.flatten(olb, 1)
    oub = torch.flatten(oub, 1)
    olb, oub = model.classifier.bounds(olb, oub)
    return olb


if __name__ == "__main__":
    x = torch.Tensor(64, 64, 3, 3)
    print(x.shape)
    # exit()
    netn = vggn.vgg19(pretrained=True)
    rewire_vgg(netn)
    lb_vgg(netn, x-1, x+1)
    a = 666