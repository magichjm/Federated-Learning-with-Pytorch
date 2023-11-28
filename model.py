import torch
import torch.nn as nn


class Model(torch.nn.Module):
    def __init__(self, inchannel=1):
        super().__init__()
        self.conv_core_1 = nn.Sequential(
            nn.Conv2d(inchannel, 64 * inchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(64 * inchannel, 64 * inchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64*inchannel),
            nn.ReLU(inplace=True)
        )
        self.conv_core_2 = nn.Sequential(
            nn.Conv2d(64 * inchannel, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(inchannel),
        )
        self.l1 = torch.nn.Linear(16 ** 2, 1)

    def forward(self, x):
        out = self.conv_core_1(x)
        out = self.conv_core_2(out)
        out = self.l1(torch.reshape(out, [out.shape[0], 1, 16 ** 2]))
        return out


# if __name__ == "__main__":
#     model = Model()
#     x = torch.zeros((10, 1, 16, 16))
#     print(model(x))
