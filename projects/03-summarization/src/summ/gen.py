import argparse
import json
import time
from pathlib import Path

import torch
from labcore import dev
from tqdm import tqdm

from summ import data


def gen_all(mdl, tok, src, d, bs=16, max_src=512, **kw):
    # longest first so the batches pad as little as possible, results put back in the original order
    order = sorted(range(len(src)), key=lambda i: -len(src[i]))
    hyp = [None] * len(src)
    for i in tqdm(range(0, len(src), bs), mininterval=60):
        ix = order[i : i + bs]
        enc = tok(
            [src[j] for j in ix], truncation=True, max_length=max_src, padding=True, return_tensors="pt"
        ).to(d)
        with torch.no_grad(), torch.autocast(d.type, dtype=torch.float16, enabled=d.type == "cuda"):
            o = mdl.generate(**enc, **kw)
        for j, t in zip(ix, tok.batch_decode(o, skip_special_tokens=True)):
            hyp[j] = t.strip()
    return hyp


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--mdl", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="cnndm", choices=list(data.sets))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--perm", action="store_true", help="use the train permutation order (for kd targets)")
    ap.add_argument("--prefix", default=None, help="short / medium / long, or none")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=142)
    ap.add_argument("--min-len", type=int, default=0)
    ap.add_argument("--lp", type=float, default=2.0)
    ap.add_argument("--nr", type=int, default=3)
    ap.add_argument("--hub-defaults", action="store_true", help="use the checkpoint's own generation config")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-src", type=int, default=512)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--runs-repo", default=None)
    a = ap.parse_args()

    d = dev()
    ds, sc, tc = data.load(a.data, a.split)
    if a.perm:
        from summ.train import order

        idx = order(len(ds))[: a.n]
    else:
        idx = data.subset(len(ds), a.n)
    sub = ds.select([int(i) for i in idx])
    src = list(sub[sc])
    inp = [data.with_prefix(s, a.prefix) for s in src] if a.prefix else src
    tok = AutoTokenizer.from_pretrained(a.mdl)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(a.mdl).to(d).eval()
    if d.type == "cuda":
        mdl.half()
    kw = (
        {}
        if a.hub_defaults
        else dict(
            num_beams=a.beams,
            max_length=a.max_len,
            min_length=a.min_len,
            length_penalty=a.lp,
            no_repeat_ngram_size=a.nr,
            early_stopping=True,
        )
    )
    if a.hub_defaults:
        kw["num_beams"] = a.beams
    t0 = time.time()
    hyp = gen_all(mdl, tok, inp, d, a.bs, a.max_src, **kw)
    sec = time.time() - t0
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "gens.jsonl", "w") as f:
        for i, h, r in zip(sub["id"], hyp, sub[tc]):
            f.write(json.dumps({"id": i, "hyp": h, "ref": r}) + "\n")
    json.dump({"args": vars(a), "sec": sec, "n": len(hyp)}, open(out / "gen.json", "w"), indent=2)
    print("done", a.name, f"{sec:.0f}s", flush=True)
    if a.runs_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.runs_repo, repo_type="dataset", exist_ok=True)
        for fn in ["gens.jsonl", "gen.json"]:
            api.upload_file(
                path_or_fileobj=out / fn,
                path_in_repo=f"{a.name}/{fn}",
                repo_id=a.runs_repo,
                repo_type="dataset",
            )


if __name__ == "__main__":
    main()
