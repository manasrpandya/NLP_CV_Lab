import cv2
import numpy as np
import torch

from bden.bf import bilat
from bden.est import immerkaer

# classical baselines. all take a float32 gray image in [0, 1] (unclipped noise) and a param dict


def interp(table, sig, keys=("d", "sig_s", "sig_r")):
    sg = [t["sigma"] for t in table]
    out = {k: float(np.interp(sig, sg, [t[k] for t in table])) for k in keys}
    if "d" in out:
        out["d"] = int(2 * round((out["d"] - 1) / 2) + 1)
    return out


def est_sigma(y, cal):
    # calibrated immerkaer estimate, 0-255 scale
    return max(cal["a"] * immerkaer(y) * 255 + cal["b"], 1.0)


def bf(y, d, sig_s, sig_r):
    return cv2.bilateralFilter(y, d, sig_r / 255, sig_s)


def bf_iter(y, d, sig_s, sig_r, n):
    for _ in range(n):
        y = cv2.bilateralFilter(y, d, sig_r / 255, sig_s)
    return y


def bf_joint(y, sig_s, sig_r, gs, k=7):
    g = cv2.GaussianBlur(y, (0, 0), gs)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))[None, None]  # noqa: E731
    with torch.no_grad():
        return bilat(t(y), sig_s, sig_r / 255, k, guide=t(g))[0, 0].numpy()


def nlm(y, h):
    u8 = np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoising(u8, None, h, 7, 21).astype(np.float32) / 255


def bm3d_(y, sig):
    import bm3d

    return bm3d.bm3d(y, sigma_psd=sig / 255).astype(np.float32)
