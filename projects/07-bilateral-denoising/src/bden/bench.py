import time
from pathlib import Path

import numpy as np
import torch
from labcore import dump

from bden import data, zoo
from bden.evaluate import cpu_fn

here = Path(__file__).resolve().parents[2]


def clock(f, n=5):
    f()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    # one 512 x 512 gray image (kodim, cropped), sigma 25. classical on one cpu thread, nets on gpu and cpu
    torch.set_num_threads(1)
    x = data.read(data.files("kodak24")[0])[:512, :512]
    y = (x + np.random.default_rng(0).normal(0, 25 / 255, x.shape)).astype(np.float32)
    out = {}
    for m in ["bf_oracle", "bf_auto", "bf_iter", "bf_joint", "nlm", "bm3d"]:
        out[m] = {"cpu": clock(lambda m=m: cpu_fn(m, y, 25), n=1 if m == "bm3d" else 5)}
        print(m, out[m], flush=True)
    t = torch.from_numpy(y)[None, None]
    for m in ["nbf", "nbf_1stage", "cnn", "dncnn"]:
        mdl = zoo.dncnn() if m == "dncnn" else zoo.ours(m)
        r = {"params": sum(p.numel() for p in mdl.parameters())}
        with torch.no_grad():
            torch.set_num_threads(4)
            r["cpu4"] = clock(lambda mdl=mdl: mdl(t))
            torch.set_num_threads(1)
            if torch.cuda.is_available():
                g, tg = mdl.cuda(), t.cuda()

                def f(g=g, tg=tg):
                    g(tg)
                    torch.cuda.synchronize()

                r["gpu"] = clock(f, 20)
        out[m] = r
        print(m, r, flush=True)
    p = here / "ckpt" / "nbf.onnx"
    if p.exists():
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        out["nbf_onnx"] = {"cpu4": clock(lambda: sess.run(None, {"x": t.numpy()}))}
    out["_note"] = "seconds per 512x512 gray image, median. cpu = 1 thread, cpu4 = 4 threads, gpu = rtx 2050"
    dump(out, here / "results" / "bench.json")


if __name__ == "__main__":
    main()
