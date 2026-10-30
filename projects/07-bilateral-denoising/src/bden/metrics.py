import numpy as np
from skimage.metrics import structural_similarity


def psnr(x, y):
    mse = np.mean((np.clip(y, 0, 1) - x) ** 2, dtype=np.float64)
    return float(10 * np.log10(1 / mse))


def ssim(x, y):
    return float(structural_similarity(x, np.clip(y, 0, 1).astype(np.float32), data_range=1.0))
