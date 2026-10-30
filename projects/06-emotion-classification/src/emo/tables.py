import re
from pathlib import Path

from labcore import dump, load

from emo.labels import ek_names

order = [
    "tfidf",
    "bigru",
    "student-hard",
    "student-kd",
    "student-int8",
    "teacher-bce",
    "teacher-asl",
    "teacher-ens",
]
site = Path("../../web/lib/emotion-classification/data.json")


def mb(k, b):
    return {"student-kd": b.get("student", {}).get("mb"), "student-hard": b.get("student", {}).get("mb"),
            "student-int8": b.get("onnx-int8", {}).get("mb"), "teacher-bce": b.get("teacher", {}).get("mb"),
            "teacher-asl": b.get("teacher", {}).get("mb")}.get(k)  # fmt: skip


def md(m, b):
    ls = ["| model | macro-f1 | micro-f1 | ekman macro-f1 | ood acc (twitter) | ece raw / cal | size (mb) |",
          "|---|---|---|---|---|---|---|"]  # fmt: skip
    for k in order:
        if k not in m:
            continue
        r = m[k]
        s = mb(k, b)
        ls.append(
            f"| {r['label']} | {r['macro_f1']:.3f} | {r['micro_f1']:.3f} | {r['ekman_macro_f1']:.3f} | "
            f"{r['ood_acc']:.3f} | {r['ece_raw']:.4f} / {r['ece_cal']:.4f} | {f'{s:.0f}' if s else ''} |"
        )
    return "\n".join(ls)


def tex(m, b):
    ls = []
    best = max(m[k]["macro_f1"] for k in m)
    for k in order:
        if k not in m:
            continue
        r = m[k]
        s = mb(k, b)
        f = f"{r['macro_f1']:.3f}"
        f = rf"\textbf{{{f}}}" if r["macro_f1"] == best else f
        ls.append(
            f"{r['label']} & {f} & {r['micro_f1']:.3f} & {r['macro_f1_at_05']:.3f} & {r['ekman_macro_f1']:.3f} & "
            f"{r['ood_acc']:.3f} & {r['ood_macro_f1']:.3f} & {f'{s:.0f}' if s else '--'} \\\\"
        )
    return "\n".join(ls) + "\n"


def tex_ece(m):
    return "".join(
        f"{m[k]['label']} & {m[k]['temp']:.2f} & {m[k]['ece_raw']:.4f} & {m[k]['ece_cal']:.4f} & "
        f"{m[k]['ece_pos_raw']:.4f} & {m[k]['ece_pos_cal']:.4f} \\\\\n"
        for k in order
        if k in m
    )


def tex_ekman(m):
    ks = [k for k in ["tfidf", "student-int8", "teacher-ens"] if k in m]
    head = " & ".join(m[k]["label"] for k in ks)
    rows = "".join(
        f"{g} & " + " & ".join(f"{m[k]['ekman_f1'][g]:.3f}" for k in ks) + " \\\\\n" for g in ek_names
    )
    return head, rows


def main():
    m = load("results/metrics.json")
    b = load("results/bench.json") if Path("results/bench.json").exists() else {}
    t = md(m, b)
    print(t)
    p = Path("README.md")
    if p.exists():
        s = p.read_text()
        s = re.sub(r"<!-- res -->.*<!-- /res -->", f"<!-- res -->\n{t}\n<!-- /res -->", s, flags=re.S)
        p.write_text(s)
    Path("report/tables").mkdir(parents=True, exist_ok=True)
    Path("report/tables/main.tex").write_text(tex(m, b))
    Path("report/tables/ece.tex").write_text(tex_ece(m))
    head, rows = tex_ekman(m)
    Path("report/tables/ekman.tex").write_text(rows)
    Path("report/tables/ekman_head.tex").write_text(head + "\n")

    dep = m["student-int8"] if "student-int8" in m else m["student-kd"]
    keys = ["label", "macro_f1", "micro_f1", "ekman_macro_f1", "ood_acc", "ece_raw", "ece_cal"]
    rows = [{"k": k, **{x: m[k][x] for x in keys}, "mb": mb(k, b)} for k in order if k in m]
    dump(
        {"temp": round(dep["temp"], 4), "th": dep["th"], "rows": [{**r, "mb": r["mb"] or 0} for r in rows]},
        site,
    )


if __name__ == "__main__":
    main()
