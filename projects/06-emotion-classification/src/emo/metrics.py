import numpy as np

grid = np.round(np.arange(0.02, 0.96, 0.01), 2)


def sig(x):
    return 1 / (1 + np.exp(-x))


def f1s(y, pred):
    # per-column f1 for multi-hot arrays
    tp = (y * pred).sum(0)
    fp = ((1 - y) * pred).sum(0)
    fn = (y * (1 - pred)).sum(0)
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def micro(y, pred):
    tp = (y * pred).sum()
    return float(2 * tp / max(2 * tp + ((1 - y) * pred).sum() + (y * (1 - pred)).sum(), 1))


def tune_th(y, p):
    # best threshold per label on val, searched independently
    pred = p[:, :, None] >= grid[None, None, :]
    yy = y[:, :, None]
    tp = (yy * pred).sum(0)
    fp = ((1 - yy) * pred).sum(0)
    fn = (yy * (1 - pred)).sum(0)
    f = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    return grid[f.argmax(1)]


def predict(p, th, top1=True):
    pred = (p >= th).astype(np.float32)
    if top1:
        # every goemotions example has at least one label, so fall back to the closest one
        none = pred.sum(1) == 0
        pred[none, (p[none] / th).argmax(1)] = 1
    return pred


def fit_temp(z, y):
    # single temperature for all labels, grid search on val bce (convex in 1/t so a fine grid is enough)
    ts = np.exp(np.linspace(np.log(0.3), np.log(5), 300))
    best, bt = np.inf, 1.0
    for t in ts:
        p = np.clip(sig(z / t), 1e-7, 1 - 1e-7)
        nll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        if nll < best:
            best, bt = nll, float(t)
    return bt


def rel_bins(p, y, nb=15):
    # reliability over every (example, label) pair
    p, y = p.ravel(), y.ravel()
    b = np.minimum((p * nb).astype(int), nb - 1)
    n = np.bincount(b, minlength=nb)
    conf = np.bincount(b, weights=p, minlength=nb) / np.maximum(n, 1)
    acc = np.bincount(b, weights=y, minlength=nb) / np.maximum(n, 1)
    return conf, acc, n


def ece(p, y, nb=15):
    conf, acc, n = rel_bins(p, y, nb)
    return float((n * np.abs(conf - acc)).sum() / n.sum())


def ece_pos(p, y, nb=15):
    # ece restricted to the predicted label slots with p > 0.05, the flat version above is dominated by
    # the huge number of easy negatives
    m = p > 0.05
    return ece(p[m], y[m], nb)
