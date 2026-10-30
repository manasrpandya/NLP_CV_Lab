import argparse
import itertools
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from labcore import dump

from bden import data
from bden.est import immerkaer, wavelet
from bden.metrics import psnr

res = Path(__file__).resolve().parents[2] / "results"
ds_ = [3, 5, 7, 9, 11]
ss_ = [0.8, 1.2, 1.6, 2.0, 2.5, 3.0, 4.0, 5.0]
rs_ = [0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]  # sig_r as a multiple of the noise sigma


def _score(args):
    sig, pairs = args
    out = {}
    for d, s, rm in itertools.product(ds_, ss_, rs_):
        r = rm * sig / 255
        out[(d, s, rm)] = np.mean([psnr(x, cv2.bilateralFilter(y, d, r, s)) for x, y in pairs])
    return sig, out


def main():
    ap = argparse.ArgumentParser(description="grid search cv2 bilateral params on the val split")
    ap.add_argument("--sigmas", type=int, nargs="+", default=list(range(5, 65, 5)))
    a = ap.parse_args()

    jobs = [(s, data.noisy("val", s, seed=1)) for s in a.sigmas]
    with Pool(8) as pool:
        scores = dict(pool.map(_score, jobs))
    table = []
    for sig in a.sigmas:
        (d, s, rm), best = max(scores[sig].items(), key=lambda kv: kv[1])
        # many windows tie once the spatial gaussian has decayed, keep the smallest one within 0.02 db
        d = min(dd for dd in ds_ if scores[sig][(dd, s, rm)] >= best - 0.02)
        table.append({"sigma": sig, "d": d, "sig_s": s, "sig_r": rm * sig, "psnr": best})
        print(table[-1])

    # sig_r (0-255 units) is close to linear in sigma, fit a line for the report
    sg = np.array([t["sigma"] for t in table], float)
    a_, b_ = np.polyfit(sg, [t["sig_r"] for t in table], 1)

    # calibrate the blind estimators: true sigma ~ a * est + b, fit on val
    cal = {}
    for name, f in (("immerkaer", immerkaer), ("wavelet", wavelet)):
        est, tru = [], []
        for sig, pairs in jobs:
            for _, y in pairs:
                est.append(f(y) * 255)
                tru.append(sig)
        est, tru = np.array(est), np.array(tru)
        k, c = np.polyfit(est, tru, 1)
        cal[name] = {
            "a": k,
            "b": c,
            "mae_raw": float(np.abs(est - tru).mean()),
            "mae_cal": float(np.abs(k * est + c - tru).mean()),
        }
        print(name, cal[name])
    dump({"table": table, "sig_r_fit": {"a": a_, "b": b_}, "est": cal}, res / "auto_fit.json")


if __name__ == "__main__":
    main()
