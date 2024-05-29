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


def mask_tiled_images(images: torch.Tensor, concat_count: int, preserved_image: int):
    assert preserved_image < concat_count, f"invalid image index {preserved_image} for concat of {concat_count} images"
    assert images.shape[2] % concat_count == 0, f"image of shape {images.shape} should have dimension divisible by {concat_count}"

    mask = torch.zeros_like(images)
    separate_image_height = images.shape[1] / concat_count

    starting_x, ending_x = [0, images.shape[2]]
    starting_y, ending_y = [(preserved_image + i) * separate_image_height for i in range(2)]

    mask[:, int(starting_y):int(ending_y), int(starting_x):int(ending_x)] = 1

    return images * mask
