import argparse
from pathlib import Path

from labcore import load

from summ.names import main, rows, xs

# latex tables for the report and the markdown table for the readme, all built from results/*.json
here = Path(__file__).resolve().parents[2]
res = here / "results"
tabs = here / "report" / "tables"


def cap(t):
    # sentence case for the report, the readme keeps the lowercase labels
    t = (
        t.replace("bart-", "BART-")
        .replace("bart ", "BART ")
        .replace("(bart-base)", "(BART-base)")
        .replace("distilbart", "DistilBART")
        .replace("pegasus", "PEGASUS")
        .replace("textrank", "TextRank")
        .replace("kd", "KD")
    )
    return t[:1].upper() + t[1:]


def f(v, d=1):
    return "--" if v is None else f"{v:.{d}f}"


def tab(head, body, spec):
    out = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(head) + r" \\", r"\midrule"]
    for r in body:
        out.append(r if r == r"\midrule" else " & ".join(r) + r" \\")
    return "\n".join(out + [r"\bottomrule", r"\end{tabular}"])


def best(m, ks, key):
    return max(m[k][key] for k in ks if k in m and key in m[k])


def tex_main(m, b):
    bk = {"ours-medium-b4": "torch-ours-b4", "ours-medium-greedy": "onnx-int8-greedy"}
    ks = [k for k in main if k in m]
    body = []
    for k in ks:
        if k == "distilbart-cnn" or k == "ours-medium-b4":
            body.append(r"\midrule")
        lab, _, pm = rows[k]
        r = m[k]
        t = b.get(bk.get(k, k), {}).get("mean_ms")
        cells = []
        for key, d in [("rouge1", 1), ("rouge2", 1), ("rougeLsum", 1), ("bertscore", 1)]:
            v = r.get(key)
            s = f(v, d)
            cells.append(rf"\textbf{{{s}}}" if v is not None and abs(v - best(m, ks, key)) < 1e-9 else s)
        c = r.get("consistency")
        cells += [f(c * 100 if c is not None else None, 0), f(r["novel_bigram"], 0), f(r["words"], 0)]
        body.append([cap(lab), str(pm) if pm else "--"] + cells + [f(t / 1000 if t else None, 2)])
    head = [
        "Method",
        "Params (M)",
        "R-1",
        "R-2",
        "R-Lsum",
        "BERTScore",
        "Faith.",
        "Novel",
        "Words",
        "CPU (s)",
    ]
    return tab(head, body, "lrcccccccc")


def tex_xsum(m):
    body = []
    for k, (lab, _, pm) in xs.items():
        if k not in m:
            continue
        r = m[k]
        c = r.get("consistency")
        body.append(
            [
                cap(lab),
                str(pm) if pm else "--",
                f(r["rouge1"]),
                f(r["rouge2"]),
                f(r["rougeLsum"]),
                f(r.get("bertscore")),
                f(c * 100 if c is not None else None, 0),
                f(r["novel_bigram"], 0),
                f(r["words"], 0),
            ]
        )
    head = ["Method", "Params (M)", "R-1", "R-2", "R-Lsum", "BERTScore", "Faith.", "Novel", "Words"]
    return tab(head, body, "lrcccccccc"[: len(head) + 1])


def tex_kd(m):
    spec = [
        ("ours-gold8k", "Gold summaries, 8k articles"),
        ("ours-kd8k", "BART-large-CNN summaries, 8k articles"),
        ("ours-medium-b4", "Gold summaries, 40k articles"),
    ]
    key = {
        "ours-gold8k": "gold8k-medium-b4",
        "ours-kd8k": "kd8k-medium-b4",
        "ours-medium-b4": "ours-medium-b4",
    }
    body = []
    for k, lab in spec:
        r = m.get(key[k])
        if r is None:
            continue
        c = r.get("consistency")
        body.append(
            [
                lab,
                f(r["rouge1"]),
                f(r["rouge2"]),
                f(r["rougeLsum"]),
                f(r.get("bertscore")),
                f(c * 100 if c is not None else None, 0),
                f(r["novel_bigram"], 0),
                f(r["words"], 0),
            ]
        )
    head = ["Training targets", "R-1", "R-2", "R-Lsum", "BERTScore", "Faith.", "Novel", "Words"]
    return tab(head, body, "lccccccc")


def tex_len(m):
    body = []
    for k, lab in [("ours-short-b4", "short"), ("ours-medium-b4", "medium"), ("ours-long-b4", "long")]:
        r = m.get(k)
        if r:
            c = r.get("consistency")
            body.append(
                [
                    lab,
                    f(r["words"], 1),
                    f(r["rouge1"]),
                    f(r["rouge2"]),
                    f(r["rougeLsum"]),
                    f(c * 100 if c is not None else None, 0),
                ]
            )
    head = ["Prefix", "Words", "R-1", "R-2", "R-Lsum", "Faith."]
    return tab(head, body, "lccccc")


def tex_onnx(o):
    body = []
    for k, lab in [("torch", "PyTorch, fp32"), ("onnx_fp32", "ONNX, fp32"), ("onnx_int8", "ONNX, int8")]:
        r = o.get(k)
        if r:
            body.append(
                [
                    lab,
                    f(r.get("mb"), 0),
                    f(r["rouge1"]),
                    f(r["rouge2"]),
                    f(r["rougeLsum"]),
                    f(r.get("same") * 100 if r.get("same") is not None else None, 0),
                ]
            )
    head = ["Model", "Size (MB)", "R-1", "R-2", "R-Lsum", "Same as PyTorch (\\%)"]
    return tab(head, body, "lccccc")


def tex_bench(b):
    names = [
        ("lead3", "Lead-3"),
        ("textrank3", "TextRank"),
        ("distilbart-cnn", "DistilBART-12-6"),
        ("bart-large-cnn", "BART-large-CNN"),
        ("pegasus-cnndm", "PEGASUS-large"),
        ("torch-ours-b4", "Ours, PyTorch, beam 4"),
        ("torch-ours-greedy", "Ours, PyTorch, greedy"),
        ("onnx-int8-greedy", "Ours, ONNX int8, greedy"),
    ]
    body = [
        [lab, f(b[k]["mean_ms"], 0), f(b[k]["median_ms"], 0), str(b[k]["n"])] for k, lab in names if k in b
    ]
    return tab(["Method", "Mean (ms)", "Median (ms)", "Articles"], body, "lrrr")


def tex_annot(a):
    cats = ["wrong number", "wrong name", "unsupported claim"]
    body = []
    for k, lab in [
        ("ours_b4", "Ours, beam 4"),
        ("ours_greedy", "Ours, greedy (browser)"),
        ("bart", "BART-large-CNN"),
    ]:
        r = a.get("systems", {}).get(k)
        if r:
            body.append([lab, str(r["n"]), str(r["clean"])] + [str(r["counts"][c]) for c in cats])
    return tab(["System", "Summaries", "No error"] + [c.capitalize() for c in cats], body, "lrrrrr")


def md(m, b, o):
    lines = [
        "| method | params | rouge-1 | rouge-2 | rouge-lsum | bertscore | faithful | novel bigrams | words |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for k in main:
        if k not in m:
            continue
        lab, _, pm = rows[k]
        r = m[k]
        c = r.get("consistency")
        lines.append(
            f"| {lab} | {pm or '-'} | {r['rouge1']:.1f} | {r['rouge2']:.1f} | {r['rougeLsum']:.1f} | "
            f"{f(r.get('bertscore'))} | {f(c * 100 if c is not None else None, 0)} | "
            f"{r['novel_bigram']:.0f} | {r['words']:.0f} |"
        )
    return "\n".join(lines)


def main_():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    m = load(res / "metrics.json")
    b = load(res / "bench.json") if (res / "bench.json").exists() else {}
    o = load(res / "onnx.json") if (res / "onnx.json").exists() else {}
    if a.md:
        print(md(m, b, o))
        return
    tabs.mkdir(parents=True, exist_ok=True)
    w = {"main.tex": tex_main(m, b), "xsum.tex": tex_xsum(m), "kd.tex": tex_kd(m), "len.tex": tex_len(m)}
    if b:
        w["bench.tex"] = tex_bench(b)
    if o:
        w["onnx.tex"] = tex_onnx(o)
    if (res / "annot.json").exists():
        w["annot.tex"] = tex_annot(load(res / "annot.json"))
    for k, v in w.items():
        (tabs / k).write_text(v + "\n")
        print("wrote", k)


if __name__ == "__main__":
    main_()
