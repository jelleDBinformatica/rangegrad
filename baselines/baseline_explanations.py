import torch
import torch.nn as nn
import numpy as np

from captum.attr import (
    GuidedGradCam,
    Occlusion,
)


def GGC_explanation(net: nn.Module, output_layer: nn.Module,
                    x: torch.Tensor,
                    y: int,
                    normalize: bool = False):
    expl = GuidedGradCam(net, output_layer)
    print("0", expl)
    explGuidedGradCam = expl.attribute(x, target=y)
    print("1", explGuidedGradCam)
    explGuidedGradCam = explGuidedGradCam.detach().cpu().numpy()
    print("2", explGuidedGradCam)
    explGuidedGradCam = explGuidedGradCam.sum(axis=0)
    print("3", explGuidedGradCam)
    if normalize:
        explGuidedGradCam = np.linalg.norm(explGuidedGradCam, axis=0)

    return explGuidedGradCam


def occlusion_explanation(net: nn.Module,
                          x: torch.Tensor,
                          y: int,
                          normalize: bool = False):
    expl = Occlusion(net)
    expl_image = expl.attribute(x, target=y, strides=(3, 25, 25), sliding_window_shapes=(3, 50, 50))
    expl_image = expl_image.cpu().detach().numpy().sum(axis=0)
    if normalize:
        expl_image = np.linalg.norm(expl_image, axis=0)

    return expl_image
