import argparse
from pathlib import Path

import numpy as np
import torch
from labcore import dump, parity

from bden import data, zoo

here = Path(__file__).resolve().parents[2]


class wrap(torch.nn.Module):
    # onnx graph returns the denoised image and the predicted maps (sig_s.., sig_r.., guide)
    def __init__(self, mdl):
        super().__init__()
        self.mdl = mdl

    def forward(self, x):
        return self.mdl(x, maps=True)


def export(mdl, p, opset=17):
    x = torch.rand(1, 1, 64, 80)
    torch.onnx.export(
        wrap(mdl).eval(),
        (x,),
        str(p),
        input_names=["x"],
        output_names=["y", "maps"],
        dynamic_axes={
            "x": {0: "n", 2: "h", 3: "w"},
            "y": {0: "n", 2: "h", 3: "w"},
            "maps": {0: "n", 2: "h", 3: "w"},
        },
        opset_version=opset,
        dynamo=False,
    )
    return p


def check(mdl, p, ims):
    errs = []
    for im in ims:
        x = torch.from_numpy(im)[None, None]
        with torch.no_grad():
            ref = [t.numpy() for t in wrap(mdl)(x)]
        ok, e = parity(ref, p, {"x": x.numpy()}, atol=1e-4)
        errs.append(max(e))
    return max(errs) <= 1e-4, max(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdl", default="nbf")
    a = ap.parse_args()
    mdl = zoo.ours(a.mdl)
    p = export(mdl, here / "ckpt" / f"{a.mdl}.onnx")
    rng = np.random.default_rng(0)
    ims = [x + rng.normal(0, 25 / 255, x.shape).astype(np.float32) for x, _ in data.noisy("set12", 25)[:3]]
    ok, err = check(mdl, p, ims)
    kb = p.stat().st_size / 1024
    print("parity", ok, err, f"{kb:.0f} kb")
    dump(
        {"mdl": a.mdl, "parity_ok": ok, "max_abs_err": err, "kb": kb, "opset": 17},
        here / "results" / "onnx.json",
    )


if __name__ == "__main__":
    main()
