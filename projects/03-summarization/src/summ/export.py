import argparse
import shutil
from pathlib import Path

import numpy as np


def to_onnx(src, out, opset=17):
    # optimum export with past kv + merged decoder, then flatten into onnx/ like transformers.js expects
    from optimum.exporters.onnx import main_export

    out = Path(out)
    tmp = out / "_tmp"
    main_export(str(src), output=tmp, task="text2text-generation-with-past", opset=opset, do_validation=False)
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    for f in tmp.iterdir():
        dst = out / "onnx" / f.name if f.suffix == ".onnx" else out / f.name
        shutil.move(str(f), dst)
    tmp.rmdir()
    # only the encoder and the merged decoder are needed, drop the rest to keep the repo small
    for n in ["decoder_model.onnx", "decoder_with_past_model.onnx"]:
        (out / "onnx" / n).unlink(missing_ok=True)
    return out / "onnx"


def quantize(p):
    # dynamic int8 for the matmuls and the embedding gather
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q = p.with_name(p.stem + "_quantized.onnx")
    quantize_dynamic(p, q, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"],
                     extra_options={"MatMulConstBOnly": True})
    return q


def mb(p):
    return Path(p).stat().st_size / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="manasrpandya/cnndm-summ")
    ap.add_argument("--out", default="runs/onnx")
    a = ap.parse_args()
    d = to_onnx(a.src, a.out)
    for n in ["encoder_model", "decoder_model_merged"]:
        q = quantize(d / f"{n}.onnx")
        print(n, f"fp32 {mb(d / f'{n}.onnx'):.1f} mb, int8 {mb(q):.1f} mb")


if __name__ == "__main__":
    main()
