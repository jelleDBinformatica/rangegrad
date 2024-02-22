import torch
import torchvision
import torchvision.transforms as T
from torchvision.models import vgg as vggn
from random import choices

from rangegrad.translation import translate_any_model

# original code by Sam Pinxteren
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
}


def transform_func(x):
    return int(x+1 in [1, 6, 7, 8, 10, 12, 21, 24, 27, 28, 33, 34])
target_transform = T.Compose([T.Lambda(transform_func)])


testset = torchvision.datasets.OxfordIIITPet(
    root='./datasets/Oxford',
    split="trainval",
    transform=data_transforms['val'],
    download=False,
    target_transform=target_transform
)

if __name__ == "__main__":
    samples = choices(range(len(testset)), k=10)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        num_workers=2,
        sampler=samples
        # shuffle=True
    )

    netn = vggn.vgg19(weights=vggn.VGG19_Weights.IMAGENET1K_V1)
    netn.eval()
    netn_translated = translate_any_model(netn)

    for i, (data, target) in enumerate(testloader):
        print(data)
        x1 = netn(data).shape
        x2 = netn_translated(data).shape
        netn_translated.set_to_lower()
        bounds = netn_translated(data)
        print("bounds", bounds)
        print(x1 == x2)
        print(target)
        exit()


