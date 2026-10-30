import torch
import torch.nn as nn
import torch.nn.functional as F

from bden.bf import bilat


def block(i, o, stride=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, stride, 1), nn.LeakyReLU(0.1), nn.Conv2d(o, o, 3, 1, 1),
                         nn.LeakyReLU(0.1))


class unet(nn.Module):
    def __init__(self, c=(16, 32, 48), out=5):
        super().__init__()
        self.e1 = block(1, c[0])
        self.e2 = block(c[0], c[1], 2)
        self.e3 = block(c[1], c[2], 2)
        self.d2 = block(c[2] + c[1], c[1])
        self.d1 = block(c[1] + c[0], c[0])
        self.head = nn.Conv2d(c[0], out, 1)

    def forward(self, x):
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(torch.cat([up(e3, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        return self.head(d1)


class nbf(nn.Module):
    # neural-guided bilateral: a small unet predicts per-pixel sig_s, sig_r (and a guide image) for one or
    # two bilateral passes. the output is always a normalised weighted average of noisy neighbours.
    def __init__(self, k=7, stages=2, guide=True, c=(16, 32, 48)):
        super().__init__()
        self.k, self.stages, self.guide = k, stages, guide
        self.net = unet(c, 2 * stages + int(guide))

    def params(self, x):
        o = self.net(x)
        ss = [0.3 + 3.2 * torch.sigmoid(o[:, 2 * i : 2 * i + 1]) for i in range(self.stages)]
        sr = [0.003 + 0.5 * torch.sigmoid(o[:, 2 * i + 1 : 2 * i + 2] - 2) for i in range(self.stages)]
        g = x + o[:, -1:] if self.guide else None
        return ss, sr, g

    def forward(self, x, maps=False):
        ss, sr, g = self.params(x)
        y = bilat(x, ss[0], sr[0], self.k, guide=g)
        for i in range(1, self.stages):
            y = bilat(y, ss[i], sr[i], self.k)
        if maps:
            return y, torch.cat(ss + sr + ([g] if g is not None else []), 1)
        return y


class cnn(nn.Module):
    # control: same unet, predicts the noise residual directly (no bilateral structure)
    def __init__(self, c=(16, 32, 48)):
        super().__init__()
        self.net = unet(c, 1)

    def forward(self, x, maps=False):
        y = x - self.net(x)
        return (y, None) if maps else y


def build(name):
    return {
        "nbf": lambda: nbf(),
        "nbf_noguide": lambda: nbf(guide=False),
        "nbf_1stage": lambda: nbf(stages=1),
        "cnn": lambda: cnn(),
    }[name]()
