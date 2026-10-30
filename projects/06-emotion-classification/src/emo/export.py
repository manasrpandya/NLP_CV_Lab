import argparse
import shutil
from pathlib import Path

import numpy as np
from labcore import dump, load, parity

from emo import metrics
from emo.labels import names


def to_onnx(mdl_dir, out, opset=17):
    # optimum export, then move things into the layout transformers.js expects (onnx/model*.onnx)
    from optimum.exporters.onnx import main_export

    out = Path(out)
    tmp = out / "_tmp"
    main_export(str(mdl_dir), output=tmp, task="text-classification", opset=opset, do_validation=False)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    for f in tmp.iterdir():
        dst = out / "onnx" / f.name if f.suffix == ".onnx" else out / f.name
        shutil.move(str(f), dst)
    tmp.rmdir()
    return out / "onnx" / "model.onnx"


def quantize(p):
    # dynamic int8: weights stored as int8, activations quantized on the fly. gather (embeddings) too.
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q = p.with_name("model_quantized.onnx")
    quantize_dynamic(p, q, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])
    return q


def to_fp16(p):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    q = p.with_name("model_fp16.onnx")
    onnx.save(convert_float_to_float16(onnx.load(p), keep_io_types=True), q)
    return q


def feeds(tok, xs):
    enc = tok(xs, padding=True, truncation=True, max_length=128, return_tensors="np")
    return {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }


def ort_logits(p, tok, xs, bs=64):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    return np.concatenate([sess.run(None, feeds(tok, xs[i : i + bs]))[0] for i in range(0, len(xs), bs)])


def torch_logits(mdl, tok, xs, bs=64):
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(xs), bs):
            f = {k: torch.tensor(v) for k, v in feeds(tok, xs[i : i + bs]).items()}
            out.append(mdl(**f).logits.numpy())
    return np.concatenate(out)


samples = [
    "thank you so much, this made my day!",
    "i can't believe they did that to [NAME], absolutely disgusting",
    "wait what? how does that even work",
    "i'm so sorry for your loss",
    "ok",
]


def card(res, m):
    r = m["student-int8"]
    return f"""---
license: mit
datasets: [google-research-datasets/go_emotions]
language: [en]
pipeline_tag: text-classification
library_name: transformers.js
tags: [onnx, emotion, multi-label, goemotions]
---

# goemotions-distil-onnx

distilroberta-base fine-tuned on GoEmotions (27 emotions + neutral, multi-label) with knowledge distillation
from a roberta-large teacher ensemble ([manasrpandya/goemotions-teacher](https://huggingface.co/manasrpandya/goemotions-teacher)).
exported to ONNX for transformers.js. `onnx/model_quantized.onnx` is int8 dynamic quantization
({res["parity"]["int8"]["mb"]:.0f} mb), `onnx/model_fp16.onnx` is for webgpu, `onnx/model.onnx` is fp32.

test set (int8 model, per-label thresholds tuned on validation): macro-f1 {r["macro_f1"]:.3f}, micro-f1
{r["micro_f1"]:.3f}, ekman macro-f1 {r["ekman_macro_f1"]:.3f}.

the outputs are logits, use sigmoid(logits / {r["temp"]:.3f}) for calibrated probabilities. the tuned
thresholds are in `calib.json`. live demo: https://nlp-cv-lab.vercel.app/p/emotion-classification,
code: https://github.com/manasrpandya/NLP_CV_Lab/tree/main/projects/06-emotion-classification
"""


def push(out, repo="manasrpandya/goemotions-distil-onnx"):
    from huggingface_hub import HfApi

    m = load("results/metrics.json")
    res = load("results/onnx.json")
    r = m["student-int8"]
    out = Path(out)
    (out / "README.md").write_text(card(res, m))
    dump({"temp": r["temp"], "th": r["th"], "labels": names}, out / "calib.json")
    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=out, commit_message="onnx export")


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from emo import data

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="manasrpandya/goemotions-student")
    ap.add_argument("--out", default="runs/onnx")
    ap.add_argument("--metrics", default="results/metrics.json")
    ap.add_argument("--run", default="student-kd")
    ap.add_argument("--push", action="store_true", help="only push an existing export to the hub")
    a = ap.parse_args()
    if a.push:
        return push(a.out)

    tok = AutoTokenizer.from_pretrained(a.src)
    mdl = AutoModelForSequenceClassification.from_pretrained(a.src).eval()
    p = to_onnx(a.src, a.out)
    pq, ph = quantize(p), to_fp16(p)

    ref = [torch_logits(mdl, tok, samples)]
    f = feeds(tok, samples)
    res = {"parity": {}}
    for k, q, tol in [("fp32", p, 1e-3), ("fp16", ph, 5e-2), ("int8", pq, 0.5)]:
        ok, errs = parity(ref, q, f, atol=tol)
        res["parity"][k] = {"ok": ok, "max_abs_err": errs[0], "mb": q.stat().st_size / 2**20}
        print(k, ok, errs, f"{q.stat().st_size / 2**20:.1f} mb")

    # f1 drop from quantization on the full test set, using the torch model's temperature + thresholds.
    # the int8 logits are also saved as their own run so evaluate.py can calibrate the deployed model.
    r = load(a.metrics)[a.run]
    ge = data.load_ge()
    xd, _ = data.load_dair("test")
    yt = ge["test"][1]
    for k, q in [("fp32", p), ("int8", pq)]:
        zt = ort_logits(q, tok, ge["test"][0])
        pred = metrics.predict(metrics.sig(zt / r["temp"]), np.array(r["th"]))
        res[f"test_macro_f1_{k}"] = float(metrics.f1s(yt, pred).mean())
        res[f"test_micro_f1_{k}"] = metrics.micro(yt, pred)
        print(k, res[f"test_macro_f1_{k}"])
        if k == "int8":
            zs = {"test": zt, "val": ort_logits(q, tok, ge["validation"][0]), "dair": ort_logits(q, tok, xd)}
            Path("runs/student-int8").mkdir(parents=True, exist_ok=True)
            np.savez_compressed("runs/student-int8/logits.npz", **zs)
    res["torch_macro_f1"] = r["macro_f1"]
    dump(res, "results/onnx.json")


if __name__ == "__main__":
    main()
