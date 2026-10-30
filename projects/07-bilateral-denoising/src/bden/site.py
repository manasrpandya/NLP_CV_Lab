from pathlib import Path

from labcore import dump, load

# compact results for the demo page (web/ can't read files outside its root on vercel)

here = Path(__file__).resolve().parents[2]
labels = {
    "noisy": ("noisy input", "-"),
    "bf_oracle": ("bilateral, tuned per sigma", "a"),
    "bf_auto": ("bilateral, blind auto params", "b"),
    "bf_iter": ("iterated bilateral", "c"),
    "bf_joint": ("joint bilateral (smoothed guide)", "c"),
    "nbf": ("neural-guided bilateral (ours)", "d"),
    "nbf_noguide": ("nbf, no guide", "d"),
    "nbf_1stage": ("nbf, one pass", "d"),
    "cnn": ("same unet, plain residual", "d"),
    "nlm": ("non-local means (cv2)", "e"),
    "bm3d": ("BM3D", "e"),
    "dncnn": ("DnCNN-B (pretrained)", "e"),
}


def main():
    fit = load(here / "results" / "auto_fit.json")
    ev = load(here / "results" / "eval.json")
    rows = []
    for k, (lbl, grp) in labels.items():
        if k not in ev:
            continue
        rows.append(
            {
                "key": k,
                "label": lbl,
                "group": grp,
                "psnr": {s: round(v["psnr"], 2) for s, v in ev[k].items()},
                "ssim": {s: round(v["ssim"], 4) for s, v in ev[k].items()},
            }
        )
    out = {
        "est": {"a": fit["est"]["immerkaer"]["a"], "b": fit["est"]["immerkaer"]["b"]},
        "auto": [{k: t[k] for k in ("sigma", "d", "sig_s", "sig_r")} for t in fit["table"]],
        "rows": rows,
    }
    b = here / "results" / "bench.json"
    if b.exists():
        out["bench"] = load(b)
    dump(out, here.parents[1] / "web" / "lib" / "bilateral-denoising" / "results.json")


if __name__ == "__main__":
    main()
