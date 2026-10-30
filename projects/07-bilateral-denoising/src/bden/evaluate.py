import argparse
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from labcore import dev, dump, load

from bden import classic, data, zoo
from bden.metrics import psnr, ssim

res = Path(__file__).resolve().parents[2] / "results"
cpu_methods = ["noisy", "bf_oracle", "bf_auto", "bf_iter", "bf_joint", "nlm", "bm3d"]
net_methods = ["nbf", "nbf_noguide", "nbf_1stage", "cnn", "dncnn"]
splits = [(s, g) for s in data.test_sets for g in data.sigmas] + [("sidd", 0)]


def cpu_fn(m, y, sig):
    # sig is the true sigma for synthetic noise, None for real noise (then everything uses the estimate)
    fit, ext = load(res / "auto_fit.json"), load(res / "tune_extra.json")["table"]
    est = classic.est_sigma(y, fit["est"]["immerkaer"])
    s = sig or est
    if m == "noisy":
        return y
    if m == "bf_oracle":
        return classic.bf(y, **classic.interp(fit["table"], s))
    if m == "bf_auto":
        return classic.bf(y, **classic.interp(fit["table"], est))
    if m == "bf_iter":
        rows = [{"sigma": t["sigma"], **t["bf_iter"]} for t in ext]
        p = classic.interp(rows, s, ("d", "sig_s", "sig_r", "n"))
        return classic.bf_iter(y, p["d"], p["sig_s"], p["sig_r"], int(round(p["n"])))
    if m == "bf_joint":
        rows = [{"sigma": t["sigma"], **t["bf_joint"]} for t in ext]
        return classic.bf_joint(y, **classic.interp(rows, s, ("sig_s", "sig_r", "gs")))
    if m == "nlm":
        rows = [{"sigma": t["sigma"], **t["nlm"]} for t in ext]
        return classic.nlm(y, classic.interp(rows, s, ("h",))["h"])
    if m == "bm3d":
        return classic.bm3d_(y, s)
    raise KeyError(m)


def _cpu_one(m, sig, xy):
    torch.set_num_threads(1)
    x, y = xy
    t0 = time.perf_counter()
    out = cpu_fn(m, y, sig or None)
    return psnr(x, out), ssim(x, out), time.perf_counter() - t0


def run_split(m, name, sig, pool):
    pairs = data.noisy(name, sig)
    if m in cpu_methods:
        r = pool.map(partial(_cpu_one, m, sig), pairs)
    else:
        d = dev()
        mdl = (zoo.dncnn() if m == "dncnn" else zoo.ours(m)).to(d)
        r = []
        with torch.no_grad():
            for x, y in pairs:
                t = torch.from_numpy(y)[None, None].to(d)
                torch.cuda.synchronize() if d.type == "cuda" else None
                t0 = time.perf_counter()
                out = mdl(t)[0, 0].float().cpu().numpy()
                r.append((psnr(x, out), ssim(x, out), time.perf_counter() - t0))
    r = np.array(r)
    return {"psnr": r[:, 0].mean(), "ssim": r[:, 1].mean(), "sec": np.median(r[:, 2]), "per": r[:, 0].round(3).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=cpu_methods)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    p = res / "eval.json"
    out = load(p) if p.exists() else {}
    with Pool(a.procs) as pool:
        for m in a.methods:
            out[m] = {}
            for name, sig in splits:
                r = run_split(m, name, sig, pool)
                out[m][f"{name}/{sig}"] = r
                print(m, name, sig, round(r["psnr"], 2), round(r["ssim"], 4), flush=True)
            dump(out, p)


if __name__ == "__main__":
    main()
