from pathlib import Path

import torch
import torch.nn as nn

from bden.model import build

here = Path(__file__).resolve().parents[2]


def dncnn():
    # pretrained blind gray dncnn (zhang et al. 2017), weights from the kair model zoo
    layers = [nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU()]
    for _ in range(18):
        layers += [nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU()]
    layers += [nn.Conv2d(64, 1, 3, 1, 1)]
    net = nn.Sequential(*layers)
    sd = torch.load(here / "data" / "dncnn_gray_blind.pth", map_location="cpu")
    net.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()})

    class res(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):
            return x - self.net(x)

    return res().eval()


def ours(name):
    mdl = build(name)
    mdl.load_state_dict(torch.load(here / "ckpt" / f"{name}.pt", map_location="cpu"))
    return mdl.eval()
