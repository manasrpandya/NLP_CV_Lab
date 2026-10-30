import cv2
import numpy as np
import torch

from bden.bf import bilat
from bden.est import immerkaer, wavelet


def _img(n=48, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:n, :n] / n
    x = 0.5 + 0.3 * np.sin(6 * xx) * np.cos(4 * yy) + 0.2 * (xx > 0.5)
    return (x + rng.normal(0, 0.05, x.shape)).astype(np.float32)


def test_matches_cv2():
    x = _img()
    for k, s, r in [(5, 1.5, 0.1), (7, 2.0, 0.08), (9, 3.0, 0.2)]:
        ref = cv2.bilateralFilter(x, k, r, s, borderType=cv2.BORDER_REFLECT_101)
        got = bilat(torch.from_numpy(x)[None, None], s, r, k=k)[0, 0].numpy()
        # cv2 uses a lookup table for the range kernel on float input, so not bit exact
        assert np.abs(ref - got).max() < 5e-3, (k, np.abs(ref - got).max())


def test_maps_and_guide():
    x = torch.from_numpy(_img())[None, None]
    s = torch.full_like(x, 2.0)
    r = torch.full_like(x, 0.1)
    assert torch.allclose(bilat(x, s, r), bilat(x, 2.0, 0.1), atol=1e-6)
    # huge sig_r turns it into a plain gaussian blur, whatever the guide is
    a = bilat(x, 2.0, 1e3, guide=torch.zeros_like(x))
    b = bilat(x, 2.0, 1e3)
    assert torch.allclose(a, b, atol=1e-5)


def test_noise_est():
    rng = np.random.default_rng(1)
    x = cv2.GaussianBlur(rng.random((256, 256)).astype(np.float32), (0, 0), 4)
    for sig in (5, 15, 25, 50):
        y = x + rng.normal(0, sig / 255, x.shape)
        for f in (immerkaer, wavelet):
            assert abs(f(y) * 255 - sig) < 0.1 * sig + 1, (f.__name__, sig, f(y) * 255)
