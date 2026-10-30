import re
from pathlib import Path

from labcore import load

from bden.evaluate import load_eval
from bden.site import labels

# latex tables for the report and the markdown table in the readme, all from results/*.json

here = Path(__file__).resolve().parents[2]
res = here / "results"
main_rows = ["noisy", "bf_oracle", "bf_auto", "bf_iter", "bf_joint", "nbf", "nlm", "bm3d", "dncnn"]
bil = ["bf_oracle", "bf_auto", "bf_iter", "bf_joint", "nbf"]
sets = [("set12", "Set12"), ("bsd68", "BSD68"), ("kodak24", "Kodak24")]
sigs = [15, 25, 50]


def cap(t):
    # sentence case for the report, the readme keeps the short lowercase labels
    t = t.replace("nbf", "NBF").replace("unet", "U-Net").replace("onnx runtime", "ONNX Runtime")
    return t[:1].upper() + t[1:]


def tex_main(ev):
    cols = [f"{d}/{s}" for d, _ in sets for s in sigs] + ["sidd/0"]
    best = {c: max(ev[m][c]["psnr"] for m in bil) for c in cols}
    out = [
        r"\begin{tabular}{ll" + "c" * len(cols) + "}",
        r"\toprule",
        " & & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{n}}}" for _, n in sets) + r" & SIDD \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-12}",
        " & Method & " + " & ".join(str(s) for _ in sets for s in sigs) + r" & real \\",
        r"\midrule",
    ]
    for m in main_rows:
        if m == "nlm":
            out.append(r"\midrule")
        lbl, grp = labels[m]
        cells = []
        for c in cols:
            v = ev[m][c]["psnr"]
            cells.append(rf"\textbf{{{v:.2f}}}" if m in bil and abs(v - best[c]) < 1e-9 else f"{v:.2f}")
        out.append(f"{grp} & {cap(lbl)} & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def tex_ssim(ev):
    out = [r"\begin{tabular}{lcccc}", r"\toprule", r"Method & 15 & 25 & 50 & SIDD \\", r"\midrule"]
    for m in main_rows:
        out.append(
            cap(labels[m][0])
            + " & "
            + " & ".join(f"{ev[m][c]['ssim']:.3f}" for c in [f"bsd68/{s}" for s in sigs] + ["sidd/0"])
            + r" \\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def tex_abl(ev):
    out = [
        r"\begin{tabular}{lrcccc}",
        r"\toprule",
        r"Variant & Params & Val (25) & BSD68 15 & 25 & 50 \\",
        r"\midrule",
    ]
    for m in ["nbf", "nbf_noguide", "nbf_1stage", "cnn"]:
        tr = load(res / f"train_{m}.json")
        out.append(
            f"{cap(labels[m][0])} & {tr['params'] / 1000:.1f}k & {tr['best_val25']:.2f} & "
            + " & ".join(f"{ev[m][f'bsd68/{s}']['psnr']:.2f}" for s in sigs)
            + r" \\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def tex_bench():
    b = load(res / "bench.json")
    rows = [("bf_oracle", "cpu"), ("bf_iter", "cpu"), ("bf_joint", "cpu"), ("nlm", "cpu"), ("bm3d", "cpu")]
    out = [r"\begin{tabular}{lrrr}", r"\toprule", r"Method & CPU (ms) & GPU (ms) & Params \\", r"\midrule"]
    for m, k in rows:
        out.append(f"{cap(labels[m][0])} & {b[m][k] * 1000:.0f} & - & - \\\\")
    for m in ["nbf", "cnn", "dncnn"]:
        out.append(
            f"{cap(labels[m][0])} & {b[m]['cpu4'] * 1000:.0f}$^*$ & {b[m]['gpu'] * 1000:.1f} & "
            f"{b[m]['params'] / 1000:.0f}k \\\\"
        )
    out.append(f"NBF, ONNX Runtime & {b['nbf_onnx']['cpu4'] * 1000:.0f}$^*$ & - & - \\\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def tex_est():
    e = load(res / "est.json")
    out = [r"\begin{tabular}{lccc}", r"\toprule", r"Estimator & 15 & 25 & 50 \\", r"\midrule"]
    for k in ("immerkaer", "wavelet"):
        out.append(
            f"{k.capitalize()} & "
            + " & ".join(f"{e[k][str(s)]['mae']:.2f} ({e[k][str(s)]['bias']:+.2f})" for s in sigs)
            + r" \\"
        )
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def md(ev):
    lines = [
        "| | method | Set12 15 | 25 | 50 | BSD68 15 | 25 | 50 | Kodak24 15 | 25 | 50 | SIDD |",
        "|---|---|" + "---:|" * 10,
    ]
    for m in main_rows:
        lbl, grp = labels[m]
        v = [f"{ev[m][f'{d}/{s}']['psnr']:.2f}" for d, _ in sets for s in sigs] + [
            f"{ev[m]['sidd/0']['psnr']:.2f}"
        ]
        if m == "nbf":
            v = [f"**{x}**" for x in v]
        lines.append(f"| {grp} | {lbl} | " + " | ".join(v) + " |")
    return "\n".join(lines)


def main():
    ev = load_eval()
    d = here / "report" / "tables"
    d.mkdir(parents=True, exist_ok=True)
    (d / "main.tex").write_text(tex_main(ev) + "\n")
    (d / "ssim.tex").write_text(tex_ssim(ev) + "\n")
    (d / "abl.tex").write_text(tex_abl(ev) + "\n")
    (d / "bench.tex").write_text(tex_bench() + "\n")
    (d / "est.tex").write_text(tex_est() + "\n")
    p = here / "README.md"
    if p.exists():
        s = p.read_text()
        s = re.sub(r"<!-- tab -->.*<!-- /tab -->", f"<!-- tab -->\n{md(ev)}\n<!-- /tab -->", s, flags=re.S)
        p.write_text(s)
    print("ok")


if __name__ == "__main__":
    main()
