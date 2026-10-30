import numpy as np
from scipy.signal import convolve2d

# blind noise level estimators, both return sigma in the units of the input

_ker = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)


def immerkaer(im):
    # immerkaer 1996, fast noise variance estimation
    h, w = im.shape
    s = np.abs(convolve2d(im.astype(np.float64), _ker, mode="valid")).sum()
    return float(s * np.sqrt(np.pi / 2) / (6 * (w - 2) * (h - 2)))


def wavelet(im):
    # donoho: median of the finest diagonal haar subband / 0.6745
    im = im.astype(np.float64)
    h, w = (im.shape[0] // 2) * 2, (im.shape[1] // 2) * 2
    a, b = im[0:h:2, 0:w:2], im[0:h:2, 1:w:2]
    c, d = im[1:h:2, 0:w:2], im[1:h:2, 1:w:2]
    hh = (a - b - c + d) / 2
    return float(np.median(np.abs(hh)) / 0.6745)
