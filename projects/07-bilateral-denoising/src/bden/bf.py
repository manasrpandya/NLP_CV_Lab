import torch
import torch.nn.functional as F


def offsets(k, circ=True):
    # (dy, dx) taps of a k x k window, optionally cut to a disc like cv2 does
    r = k // 2
    out = [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
    if circ:
        out = [(dy, dx) for dy, dx in out if dy * dy + dx * dx <= r * r]
    return out


def shifts(x, k, taps):
    # stack of shifted copies (n, taps, h, w). plain slicing instead of F.unfold so the onnx graph is
    # just Slice + Concat, which every runtime (incl. ort-web webgpu) handles
    r = k // 2
    h, w = x.shape[-2:]
    xp = F.pad(x, (r, r, r, r), mode="reflect")
    return torch.cat([xp[:, :, r + dy : r + dy + h, r + dx : r + dx + w] for dy, dx in taps], 1)


def bilat(x, sig_s, sig_r, k=7, guide=None, circ=True):
    # x (n,1,h,w). sig_s / sig_r are floats or per-pixel maps (n,1,h,w), in pixels and intensity units.
    # guide: image used for the range weights (joint / cross bilateral), defaults to x itself
    taps = offsets(k, circ)
    g = x if guide is None else guide
    xs = shifts(x, k, taps)
    gs = xs if guide is None else shifts(g, k, taps)
    d2 = torch.tensor([dy * dy + dx * dx for dy, dx in taps], dtype=x.dtype, device=x.device)
    d2 = d2.view(1, -1, 1, 1)
    sig_s = torch.as_tensor(sig_s, dtype=x.dtype, device=x.device)
    sig_r = torch.as_tensor(sig_r, dtype=x.dtype, device=x.device)
    logw = -d2 / (2 * sig_s**2) - (gs - g) ** 2 / (2 * sig_r**2)
    w = torch.softmax(logw, 1)  # same as exp / sum but numerically safe for tiny sig_r
    return (w * xs).sum(1, keepdim=True)
