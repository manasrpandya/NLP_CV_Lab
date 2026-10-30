import torch
import torch.nn.functional as F


def asl(x, y, gn=4.0, gp=0.0, clip=0.05, eps=1e-8):
    # asymmetric loss (ridnik et al. 2021): focus down-weights easy negatives, clip shifts them to zero
    x = x.float()
    p = torch.sigmoid(x)
    pn = (1 - p + clip).clamp(max=1) if clip > 0 else 1 - p
    los = y * torch.log(p.clamp(min=eps)) + (1 - y) * torch.log(pn.clamp(min=eps))
    pt = p * y + pn * (1 - y)
    w = torch.pow(1 - pt, gp * y + gn * (1 - y))
    return -(los * w).mean()


def bce(x, y):
    return F.binary_cross_entropy_with_logits(x.float(), y)


def kd(x, soft_logits, t=2.0):
    # per-label binary distillation on temperature-softened teacher probs
    q = torch.sigmoid(soft_logits.float() / t)
    return F.binary_cross_entropy_with_logits(x.float() / t, q) * t * t
