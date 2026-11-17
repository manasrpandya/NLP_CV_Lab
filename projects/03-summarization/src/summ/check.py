import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from labcore import dump
from tqdm import tqdm

from summ import data
from summ.export import Ort, mb
from summ.metrics import rouge


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="manasrpandya/cnndm-summ")
    ap.add_argument("--trimmed", default="runs/trimmed")
    ap.add_argument("--onnx", default="runs/onnx")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--res", default="results/onnx.json")
    a = ap.parse_args()

    # the deploy folder gets the trimmed tokenizer files, the exporter wrote the full-vocab ones
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
        shutil.copy(Path(a.trimmed) / f, Path(a.onnx) / f)

    ds, sc, tc = data.load("cnndm", "test")
    idx = data.subset(len(ds), 2000)[: a.n]
    sub = ds.select([int(i) for i in idx])
    xs = [data.with_prefix(x, "medium") for x in sub[sc]]
    ref = list(sub[tc])

    tok = AutoTokenizer.from_pretrained(a.src)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(a.src).eval()
    ttok = AutoTokenizer.from_pretrained(a.trimmed)
    keep = np.load(Path(a.trimmed) / "keep.npy")
    kw = dict(num_beams=1, do_sample=False, max_length=142, no_repeat_ngram_size=3)

    hyp = {"torch": [], "onnx_fp32": [], "onnx_int8": []}
    o32, o8 = Ort(Path(a.onnx) / "onnx", q=False), Ort(Path(a.onnx) / "onnx", q=True)
    infl = []
    for x in tqdm(xs, mininterval=30):
        e = tok(x, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            hyp["torch"].append(tok.decode(mdl.generate(**e, **kw)[0], skip_special_tokens=True).strip())
        et = ttok(x, truncation=True, max_length=512, return_tensors="np")
        ids, mask = et["input_ids"].astype(np.int64), et["attention_mask"].astype(np.int64)
        infl.append(len(tok(x)["input_ids"]) and len(ttok(x)["input_ids"]) / len(tok(x)["input_ids"]))
        for k, o in [("onnx_fp32", o32), ("onnx_int8", o8)]:
            hyp[k].append(ttok.decode(o.gen(ids, mask), skip_special_tokens=True).strip())

    d = Path(a.onnx) / "onnx"
    sizes = {
        "torch": mdl.num_parameters() * 4 / 2**20,
        "onnx_fp32": mb(d / "encoder_model.onnx") + mb(d / "decoder_model_merged.onnx"),
        "onnx_int8": mb(d / "encoder_model_quantized.onnx") + mb(d / "decoder_model_merged_quantized.onnx"),
    }
    out = {
        "n": a.n,
        "vocab": len(keep),
        "token_inflation": float(np.mean(infl)),
        "params_full_m": mdl.num_parameters() / 1e6,
        "params_trimmed_m": None,
    }
    for k, h in hyp.items():
        r = rouge(ref, h).mean(0)
        out[k] = {
            "rouge1": float(r[0]),
            "rouge2": float(r[1]),
            "rougeLsum": float(r[2]),
            "mb": sizes[k],
            "same": float(np.mean([x == y for x, y in zip(h, hyp["torch"])])),
            "words": float(np.mean([len(x.split()) for x in h])),
        }
        print(k, out[k], flush=True)
    dump(out, a.res)
    dump({"torch": hyp["torch"], "onnx_int8": hyp["onnx_int8"]}, Path(a.onnx).parent / "check_gens.json")


if __name__ == "__main__":
    main()
