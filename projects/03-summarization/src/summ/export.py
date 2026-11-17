import argparse
import shutil
from pathlib import Path


def to_onnx(src, out, opset=17):
    # optimum export without the merge step, so the decoders can be quantized first and merged afterwards
    from optimum.exporters.onnx import main_export

    out = Path(out)
    tmp = out / "_tmp"
    main_export(
        str(src),
        output=tmp,
        task="text2text-generation-with-past",
        opset=opset,
        do_validation=False,
        no_post_process=True,
    )
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    for f in tmp.iterdir():
        dst = out / "onnx" / f.name if f.suffix == ".onnx" else out / f.name
        shutil.move(str(f), dst)
    tmp.rmdir()
    return out / "onnx"


def quantize(p):
    # dynamic 8-bit quantization of the matmuls and the embedding gather
    from onnxruntime.quantization import QuantType, quantize_dynamic

    q = p.with_name(p.stem + "_quantized.onnx")
    # unsigned weights with asymmetric activations lost about 1 rouge point less than signed ones in my tests
    quantize_dynamic(
        p,
        q,
        weight_type=QuantType.QUInt8,
        op_types_to_quantize=["MatMul", "Gather"],
        extra_options={"ActivationSymmetric": False},
    )
    return q


def merge(d, suffix=""):
    from optimum.onnx import merge_decoders

    a, b = d / f"decoder_model{suffix}.onnx", d / f"decoder_with_past_model{suffix}.onnx"
    out = d / f"decoder_model_merged{suffix}.onnx"
    merge_decoders(a, b, save_path=out, strict=False)
    return out


def build(src, out):
    # the layout transformers.js expects: onnx/encoder_model(.onnx|_quantized.onnx), decoder_model_merged(...)
    d = to_onnx(src, out)
    for n in ["encoder_model", "decoder_model", "decoder_with_past_model"]:
        quantize(d / f"{n}.onnx")
    merge(d)
    merge(d, "_quantized")
    for n in [
        "decoder_model",
        "decoder_with_past_model",
        "decoder_model_quantized",
        "decoder_with_past_model_quantized",
    ]:
        (d / f"{n}.onnx").unlink(missing_ok=True)
    return d


class Ort:
    # greedy decoding on the exported encoder + merged decoder, same loop transformers.js runs in the browser
    def __init__(self, d, q=True):
        import onnxruntime as ort

        d = Path(d)
        sfx = q if isinstance(q, str) else "_quantized" if q else ""
        o = ort.SessionOptions()
        o.intra_op_num_threads = 4
        self.enc = ort.InferenceSession(
            str(d / f"encoder_model{sfx}.onnx"), o, providers=["CPUExecutionProvider"]
        )
        self.dec = ort.InferenceSession(
            str(d / f"decoder_model_merged{sfx}.onnx"), o, providers=["CPUExecutionProvider"]
        )
        pk = [i for i in self.dec.get_inputs() if i.name.startswith("past_key_values")]
        self.pnames = [i.name for i in pk]
        self.nh, self.hd = pk[0].shape[1], pk[0].shape[3]

    def encode(self, ids, mask):
        return self.enc.run(None, {"input_ids": ids, "attention_mask": mask})[0]

    def gen(self, ids, mask, max_len=142, nr=3, start=2, bos=0, eos=2):
        import numpy as np

        h = self.encode(ids, mask)
        seq = [start]
        past = {n: np.zeros((1, self.nh, 0, self.hd), np.float32) for n in self.pnames}
        enc_past = {}
        for step in range(max_len - 1):
            first = step == 0
            f = {
                "encoder_attention_mask": mask,
                "encoder_hidden_states": h,
                "input_ids": np.array([[seq[-1]]], np.int64) if not first else np.array([seq], np.int64),
                "use_cache_branch": np.array([not first]),
            }
            f.update(past)
            outs = self.dec.run(None, f)
            z = outs[0][0, -1].copy()
            names = [o.name for o in self.dec.get_outputs()]
            for n, v in zip(names[1:], outs[1:]):
                k = "past_key_values." + n[len("present.") :]
                if ".encoder." in n:
                    if first:
                        enc_past[k] = v
                    past[k] = enc_past[k]
                else:
                    past[k] = v
            if len(seq) == 1:
                t = bos
            else:
                if nr and len(seq) >= nr:
                    pre = tuple(seq[-(nr - 1) :])
                    for i in range(len(seq) - nr + 1):
                        if tuple(seq[i : i + nr - 1]) == pre:
                            z[seq[i + nr - 1]] = -1e9
                t = int(z.argmax())
            seq.append(t)
            if t == eos:
                break
        return seq


def mb(p):
    return Path(p).stat().st_size / 2**20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="manasrpandya/cnndm-summ")
    ap.add_argument("--out", default="runs/onnx")
    a = ap.parse_args()
    d = build(a.src, a.out)
    for f in sorted(d.glob("*.onnx")):
        print(f.name, f"{mb(f):.1f} mb")


if __name__ == "__main__":
    main()
