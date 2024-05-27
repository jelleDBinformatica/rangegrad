import torch

from typing import List


def tile_images(images: List[torch.Tensor]):
    original_size = None
    full_image = None
    first_loop = True
    for image in images:
        if original_size is None:
            original_size = image.shape
            full_image = image
        assert original_size == image.shape, f"given a list of incompatible shapes {original_size} and {image.shape}"
        if not first_loop:
            full_image = torch.cat((full_image, image), dim=2)
        first_loop = False
    return full_image
