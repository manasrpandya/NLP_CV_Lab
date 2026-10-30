import argparse
import shutil
from pathlib import Path

import numpy as np
from labcore import dump, load, parity

from emo import metrics


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


def main():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from emo import data

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="manasrpandya/goemotions-student")
    ap.add_argument("--out", default="runs/onnx")
    ap.add_argument("--metrics", default="results/metrics.json")
    ap.add_argument("--run", default="student-kd")
    a = ap.parse_args()

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

    # f1 drop from quantization on the full test set, using the torch model's temperature + thresholds
    r = load(a.metrics)[a.run]
    ge = data.load_ge()
    yt = ge["test"][1]
    for k, q in [("fp32", p), ("int8", pq)]:
        pt = metrics.sig(ort_logits(q, tok, ge["test"][0]) / r["temp"])
        pred = metrics.predict(pt, np.array(r["th"]))
        res[f"test_macro_f1_{k}"] = float(metrics.f1s(yt, pred).mean())
        res[f"test_micro_f1_{k}"] = metrics.micro(yt, pred)
        print(k, res[f"test_macro_f1_{k}"])
    res["torch_macro_f1"] = r["macro_f1"]
    dump(res, "results/onnx.json")


if __name__ == "__main__":
    main()
