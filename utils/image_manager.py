import matplotlib.pyplot as plt
import os
from typing import List, Optional

import torch
import numpy as np


class ImageManager:
    def __init__(self,
                 directory: str,
                 normalize_image: bool = True):
        self.base_dir = directory
        self.normalize_image = normalize_image
        self.prepare_dir("")

    def _normalize(self, image: torch.Tensor):
        """
        scales values in a tensor to a range of (0, 1)
        :param image: tensor to be used as an image
        :return:
        """
        image = np.linalg.norm(image, axis=0)
        image -= image.min()
        image /= image.max()
        return image

    def prepare_dir(self, dir_name: str):
        """
        creates directory if not present
        :param dir_name:
        :return:
        """
        full_name = self.base_dir + "/" + dir_name
        if os.path.isdir(full_name):
            return
        os.mkdir(full_name)

    def save_image(self,
                   image_tensor: torch.Tensor,
                   directory: str,
                   filename: str,
                   title: str = "",
                   normalize: Optional[bool] = None):
        if normalize is None:
            normalize = self.normalize_image
        image = image_tensor.cpu()
        self.prepare_dir(directory)
        filename = '/'.join([self.base_dir, directory, filename])
        channels = image.shape[0]
        if channels == 1:
            image = torch.tile(image, (3, 1, 1))

        image = image.permute(1, 2, 0)
        if normalize:
            image = self._normalize(image)
        # plt.imshow(image, cmap=plt.cm.Reds)
        plt.axis('off')
        plt.imshow(image)
        plt.title(title)

        # TODO: set title
        plt.savefig(filename)
        plt.close()

    def save_image_batch(self,
                         images: List[torch.Tensor],
                         directory: str,
                         base_filename: str,
                         title: str = "",
                         normalize: Optional[bool] = None):
        for i, image in enumerate(images):
            new_filename = str(i) + "_" + base_filename
            self.save_image(image, directory, new_filename, title, normalize)


if __name__ == "__main__":
    t = ImageManager("./images")
    l = [
        torch.rand((1, 200, 300)) for _ in range(5)
    ]
    t.save_image_batch(l, "temp", "temp", title="temp title")




