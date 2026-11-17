import argparse
import json
from pathlib import Path

import numpy as np
from labcore import dump, load

from summ import data
from summ.metrics import boot, novel, rouge
from summ.text import words


def read(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines()]


def srcs_for(name, ids):
    # the source article of every id, looked up in the test split
    dsn = "xsum" if name.startswith("xsum") else "cnndm"
    ds, sc, _ = data.load(dsn, "test")
    want = set(ids)
    m = {}
    for r in ds:
        if r["id"] in want:
            m[r["id"]] = r[sc]
            if len(m) == len(want):
                break
    return [m[i] for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--res", default="results/metrics.json")
    ap.add_argument("--nli-n", type=int, default=500)
    ap.add_argument("--no-nli", action="store_true")
    ap.add_argument("--no-bs", action="store_true")
    ap.add_argument("--bs-model", default="roberta-large")
    a = ap.parse_args()

    res = load(a.res) if Path(a.res).exists() else {}
    nli_m = None
    if not a.no_nli:
        from summ import nli

        d = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        nli_m = nli.load(d)
    for name in a.names:
        rows = read(Path(a.runs) / name / "gens.jsonl")
        ids = [r["id"] for r in rows]
        hyp, ref = [r["hyp"] for r in rows], [r["ref"] for r in rows]
        src = srcs_for(name, ids)
        r = rouge(ref, hyp)
        o = {"n": len(rows)}
        for i, k in enumerate(["rouge1", "rouge2", "rougeLsum"]):
            o[k] = float(r[:, i].mean())
            o[k + "_ci"] = boot(r[:, i], 500)
        o["words"] = float(np.mean([len(words(h)) for h in hyp]))
        o["ref_words"] = float(np.mean([len(words(x)) for x in ref]))
        nv = [novel(s, h) for s, h in zip(src, hyp)]
        o["novel_bigram"] = float(np.mean(nv) * 100)
        o["novel_bigram_ref"] = float(np.mean([novel(s, x) for s, x in zip(src, ref)]) * 100)
        if nli_m is not None:
            n = min(a.nli_n, len(rows))
            c = nli.score(*nli_m, src[:n], hyp[:n])
            o["consistency"] = float(np.mean(c))
            o["consistency_n"] = n
            o["consistency_ref"] = float(np.mean(nli.score(*nli_m, src[:n], ref[:n])))
        if not a.no_bs:
            from bert_score import score as bs

            _, _, f = bs(hyp, ref, lang="en", model_type=a.bs_model, batch_size=32, verbose=False)
            o["bertscore"] = float(f.mean() * 100)
        gj = Path(a.runs) / name / "gen.json"
        if gj.exists():
            o["gen_sec"] = load(gj)["sec"]
        res[name] = o
        print(name, {k: round(v, 2) for k, v in o.items() if isinstance(v, float)}, flush=True)
        dump(res, a.res)


if __name__ == "__main__":
    main()
