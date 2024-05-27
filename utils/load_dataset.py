import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, IterableDataset
import datasets as ds

from torchvision.datasets import ImageNet

import matplotlib.pyplot as plt
from random import choices

data_transforms = {
    'train': T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'inverse': T.Compose([
        T.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        T.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
    ]),
    'tiling': T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
}


class ImagenetSampleDataset(Dataset):
    def __init__(self, streamed_dataset, transform):
        self.streamed_dataset = streamed_dataset
        self.transform = transform

    def __len__(self):
        return len(self.streamed_dataset["image"])

    def __getitem__(self, idx):

        if idx >= len(self):
            return None

        im = self.streamed_dataset['image'][idx]
        label = self.streamed_dataset['label'][idx]
        im = T.ToTensor()(im)

        if im.shape[0] == 1:
            im = im.tile((3, 1, 1))

        im = T.ToPILImage()(im)

        return (
            self.transform(im),
            label

        )
        # return {
        #     'image': self.transform(self.streamed_dataset['image'][idx]),
        #     'label': self.streamed_dataset['label'][idx],
        # }

    # def __iter__(self):
    #     for idx in range(len(self)):
    #         yield (self[idx][k] for k in self.streamed_dataset.keys())


def retrieve_dataset_sample(dataset: ds.IterableDataset, amount: int):
    for batch in dataset.iter(amount):
        return batch


def prepare_dataset_stream():
    dataset = ds.load_dataset("imagenet-1k", split="validation", streaming=True, trust_remote_code=True)
    return dataset


def setup_imagenet_sample(sample_size: int, transforms):
    """
    streams the imagenet dataset from hugging face and stores a sample of it
    :param sample_size:
    :param transforms:
    :return:
    """
    ds = prepare_dataset_stream()
    sample = retrieve_dataset_sample(ds, sample_size)
    result = ImagenetSampleDataset(sample, transforms)
    return result


if __name__ == "__main__":

    bleh = setup_imagenet_sample(100, data_transforms["val"])

    samples = choices(range(len(bleh)), k=100)
    testloader = torch.utils.data.DataLoader(
        bleh,
        batch_size=1,
        num_workers=1   ,
        sampler=samples
        # shuffle=True
    )

    for x, label in testloader:
        print(label)
        # plt.imshow(x.permute((1, 2, 0)))
        # plt.show()

