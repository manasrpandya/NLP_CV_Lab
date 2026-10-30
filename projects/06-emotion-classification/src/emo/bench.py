import argparse
import time
from pathlib import Path

import numpy as np
import torch
from labcore import dump

from emo import data
from emo.export import feeds


def lat(fn, xs, warm=5):
    # median single-sentence latency in ms
    for x in xs[:warm]:
        fn([x])
    ts = []
    for x in xs:
        t0 = time.perf_counter()
        fn([x])
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


def dir_mb(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*.safetensors")) / 2**20


def main():
    import onnxruntime as ort
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--onnx", default="runs/onnx/onnx")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    xs = data.load_ge()["test"][0][: a.n]
    res = {"threads": a.threads, "n": a.n}

    for k, repo in [
        ("teacher", "manasrpandya/goemotions-teacher"),
        ("student", "manasrpandya/goemotions-student"),
    ]:
        p = snapshot_download(repo)
        tok = AutoTokenizer.from_pretrained(p)
        mdl = AutoModelForSequenceClassification.from_pretrained(p).eval()

        def fn(b, tok=tok, mdl=mdl):
            with torch.no_grad():
                mdl(**{k: torch.tensor(v) for k, v in feeds(tok, b).items()})

        res[k] = {
            "mb": dir_mb(p),
            "params_m": sum(q.numel() for q in mdl.parameters()) / 1e6,
            "ms": lat(fn, xs),
        }
        print(k, res[k])
        del mdl

    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    for k, f in [("onnx-fp32", "model.onnx"), ("onnx-int8", "model_quantized.onnx")]:
        p = Path(a.onnx) / f
        sess = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
        res[k] = {"mb": p.stat().st_size / 2**20, "ms": lat(lambda b, s=sess: s.run(None, feeds(tok, b)), xs)}
        print(k, res[k])
    dump(res, "results/bench.json")


if __name__ == "__main__":
    main()
