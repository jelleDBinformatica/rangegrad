import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(4, 4),
            nn.ReLU(),
            nn.Linear(4, 20),
            # nn.Unflatten(0, (1, 4, 5))
        )
        self.seq2 = nn.Sequential(
            # nn.Conv2d(1, 1, 2),
            # nn.Flatten(),
            nn.Linear(20, 4),
            nn.ReLU()
        )
        self.sm = nn.Softmax()

    def forward(self, x):
        y = self.seq(x)
        y = self.seq2(y)
        y = self.sm(y)
        return y


if __name__ == "__main__":
    x = torch.Tensor([1, 1, 1, 1])
    cm = CustomModel()
    y = cm(x)
    print(y)