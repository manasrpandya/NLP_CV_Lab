import itertools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from labcore import dump, load

from bden import classic, data
from bden.metrics import psnr

res = Path(__file__).resolve().parents[2] / "results"
sigmas = [5, 10, 15, 25, 50]


def _grid(fn, grid, pairs):
    sc = {g: np.mean([psnr(x, fn(y, *g)) for x, y in pairs]) for g in grid}
    g = max(sc, key=sc.get)
    return list(g), float(sc[g])


def _one(sig):
    torch_threads()
    pairs = data.noisy("val", sig, seed=1)[:12]
    base = classic.interp(load(res / "auto_fit.json")["table"], sig)
    out = {}
    g = itertools.product([2, 3, 4], [1.0, 1.5, 2.0], [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    p, s = _grid(lambda y, n, ss, rm: classic.bf_iter(y, base["d"], ss, rm * sig, n), list(g), pairs)
    out["bf_iter"] = {"d": base["d"], "n": p[0], "sig_s": p[1], "sig_r": p[2] * sig, "val": s}
    g = itertools.product([0.4, 0.55, 0.7, 1.0], [1.0, 1.5, 2.0, 2.5, 3.0, 3.5], [0.25, 0.5, 0.75, 1.0, 1.5])
    p, s = _grid(lambda y, gs, ss, rm: classic.bf_joint(y, ss, rm * sig, gs), list(g), pairs)
    out["bf_joint"] = {"gs": p[0], "sig_s": p[1], "sig_r": p[2] * sig, "val": s}
    p, s = _grid(lambda y, hm: classic.nlm(y, hm * sig), [(m,) for m in [0.6, 0.8, 1.0, 1.2, 1.5]], pairs)
    out["nlm"] = {"h": p[0] * sig, "val": s}
    print(sig, out, flush=True)
    return {"sigma": sig, **out}


def torch_threads():
    import torch

    torch.set_num_threads(1)


def main():
    with Pool(5) as pool:
        dump({"table": pool.map(_one, sigmas)}, res / "tune_extra.json")


if __name__ == "__main__":
    main()
