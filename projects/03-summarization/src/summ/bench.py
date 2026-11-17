import argparse
import time
from pathlib import Path

import numpy as np
import torch
from labcore import dump, load

from summ import data
from summ.extractive import lead, textrank


def ms(f, xs, warm=1):
    for x in xs[:warm]:
        f(x)
    t = []
    for x in xs:
        t0 = time.perf_counter()
        f(x)
        t.append(time.perf_counter() - t0)
    return {"mean_ms": float(np.mean(t) * 1000), "median_ms": float(np.median(t) * 1000), "n": len(t)}


def torch_fn(name, kw, prefix=None, threads=4):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(name).eval()

    def f(x):
        enc = tok(
            data.with_prefix(x, prefix) if prefix else x, truncation=True, max_length=512, return_tensors="pt"
        )
        with torch.no_grad():
            return mdl.generate(**enc, **kw)

    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--onnx", default="runs/onnx/onnx")
    ap.add_argument("--src", default="manasrpandya/cnndm-summ")
    ap.add_argument("--trimmed", default="runs/trimmed")
    ap.add_argument("--res", default="results/bench.json")
    ap.add_argument("--skip", nargs="*", default=[])
    a = ap.parse_args()

    ds, sc, _ = data.load("cnndm", "test")
    xs = [ds[int(i)][sc] for i in data.subset(len(ds), 2000)[: a.n]]
    res = load(a.res) if Path(a.res).exists() else {"threads": 4}
    dump(res, a.res)
    res["lead3"] = ms(lambda x: lead(x, 3), xs, 0)
    res["textrank3"] = ms(lambda x: textrank(x, 3), xs, 0)
    big = dict(num_beams=4, max_length=142, no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=True)
    runs = {
        "distilbart-cnn": (
            "sshleifer/distilbart-cnn-12-6",
            dict(num_beams=4, max_length=142, early_stopping=True),
        ),
        "bart-large-cnn": (
            "facebook/bart-large-cnn",
            dict(num_beams=4, max_length=142, min_length=56, **{"early_stopping": True}),
        ),
        "pegasus-cnndm": ("google/pegasus-cnn_dailymail", dict(num_beams=4, max_length=128)),
    }
    for k, (m, kw) in runs.items():
        if k not in a.skip and k not in res:
            res[k] = ms(torch_fn(m, kw), xs[: max(5, a.n // 4)])
            print(k, res[k], flush=True)
            dump(res, a.res)
    for k, kw in {
        "ours-b4": big,
        "ours-greedy": dict(num_beams=1, max_length=142, no_repeat_ngram_size=3),
    }.items():
        res["torch-" + k] = ms(torch_fn(a.src, kw, "medium"), xs)
        print(k, res["torch-" + k], flush=True)
    from transformers import AutoTokenizer

    from summ.export import Ort

    tok = AutoTokenizer.from_pretrained(a.trimmed)
    o = Ort(a.onnx, q=True)

    def f(x):
        e = tok(data.with_prefix(x, "medium"), truncation=True, max_length=512, return_tensors="np")
        return o.gen(e["input_ids"].astype(np.int64), e["attention_mask"].astype(np.int64))

    res["onnx-int8-greedy"] = ms(f, xs)
    print(res["onnx-int8-greedy"])
    dump(res, a.res)


if __name__ == "__main__":
    main()
