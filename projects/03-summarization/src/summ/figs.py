import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from labcore import load  # noqa: E402

from summ.names import main, rows  # noqa: E402
from summ.tables import cap  # noqa: E402
from summ.text import words  # noqa: E402

here = Path(__file__).resolve().parents[2]
out = here / "report" / "figs"
res = here / "results"
runs = here / "runs"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
        "font.family": "serif",
    }
)
col = {"extractive": "#888888", "pretrained": "#3b6ea5", "ours": "#c2410c"}


def lens(name):
    return [len(words(json.loads(x)["hyp"])) for x in (runs / name / "gens.jsonl").read_text().splitlines()]


def rouge_cost():
    m, b = load(res / "metrics.json"), load(res / "bench.json")
    bk = {
        "lead3": "lead3",
        "textrank3": "textrank3",
        "distilbart-cnn": "distilbart-cnn",
        "pegasus-cnndm": "pegasus-cnndm",
        "bart-large-cnn": "bart-large-cnn",
        "ours-medium-b4": "torch-ours-b4",
        "ours-medium-greedy": "onnx-int8-greedy",
    }
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for k in main:
        if k not in m or bk[k] not in b:
            continue
        lab, grp, pm = rows[k]
        ax[0].scatter(b[bk[k]]["mean_ms"] / 1000, m[k]["rouge2"], c=col[grp], s=22)
        ax[0].annotate(
            cap(lab),
            (b[bk[k]]["mean_ms"] / 1000, m[k]["rouge2"]),
            fontsize=6.5,
            xytext=(4, -3),
            textcoords="offset points",
        )
        if pm:
            ax[1].scatter(pm, m[k]["rouge2"], c=col[grp], s=22)
            ax[1].annotate(
                cap(lab), (pm, m[k]["rouge2"]), fontsize=6.5, xytext=(4, -3), textcoords="offset points"
            )
    ax[0].set_xscale("log")
    ax[0].set_xlabel("CPU time per summary (s, 4 threads)")
    ax[0].set_ylabel("ROUGE-2")
    ax[1].set_xlabel("Parameters (millions)")
    ax[1].set_xlim(80, 780)
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "rouge_cost.pdf")


def faith():
    m = load(res / "metrics.json")
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for k in main + ["xref"]:
        if k == "xref":
            r = m[main[-1]]
            ax.scatter(r["novel_bigram_ref"], r["consistency_ref"] * 100, marker="*", c="k", s=40)
            ax.annotate(
                "Human highlights",
                (r["novel_bigram_ref"], r["consistency_ref"] * 100),
                fontsize=6.5,
                xytext=(-60, 4),
                textcoords="offset points",
            )
            continue
        if k not in m or "consistency" not in m[k]:
            continue
        lab, grp, _ = rows[k]
        ax.scatter(m[k]["novel_bigram"], m[k]["consistency"] * 100, c=col[grp], s=22)
        ax.annotate(
            cap(lab),
            (m[k]["novel_bigram"], m[k]["consistency"] * 100),
            fontsize=6.5,
            xytext=(4, -3),
            textcoords="offset points",
        )
    ax.set_xlabel("Novel bigrams (%)")
    ax.set_ylabel("Faithfulness (%)")
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out / "faith.pdf")


def length():
    m = load(res / "metrics.json")
    ks = ["ours-short-b4", "ours-medium-b4", "ours-long-b4"]
    cs = load(res / "buckets.json")["cuts"]
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw={"width_ratios": [1.2, 1]})
    ax[0].boxplot(
        [lens(k) for k in ks]
        + [
            [len(words(json.loads(x)["ref"])) for x in (runs / ks[1] / "gens.jsonl").read_text().splitlines()]
        ],
        showfliers=False,
        widths=0.55,
        medianprops={"color": col["ours"]},
    )
    ax[0].set_xticks([1, 2, 3, 4], ["short", "medium", "long", "Human"])
    ax[0].set_ylabel("Words per summary")
    ax[0].set_xlabel("Length prefix")
    for c in cs:
        ax[0].axhline(c, c="#bbbbbb", lw=0.6, ls="--")
    x = np.arange(3)
    for i, (key, lab) in enumerate(
        [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeLsum", "ROUGE-Lsum")]
    ):
        ax[1].bar(x + (i - 1) * 0.27, [m[k][key] for k in ks], 0.27, label=lab)
    ax[1].set_xticks(x, ["short", "medium", "long"])
    ax[1].set_xlabel("Length prefix")
    ax[1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "length.pdf")


def errors():
    # one summary with the hand-marked spans, drawn as highlighted text. data: results/annot.json
    a = load(res / "annot.json")["example"]
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    ax.axis("off")
    cols = {"wrong number": "#f4a261", "wrong name": "#e76f51", "unsupported claim": "#e9c46a"}
    y = 0.95
    ax.text(0, y, "Source (start of article)", fontsize=7, weight="bold", va="top", transform=ax.transAxes)
    y -= 0.09
    for ln in textwrap.wrap(a["source"], 125):
        ax.text(0, y, ln, fontsize=6.8, va="top", transform=ax.transAxes, color="#444444")
        y -= 0.075
    y -= 0.05
    ax.text(0, y, "Summary", fontsize=7, weight="bold", va="top", transform=ax.transAxes)
    y -= 0.09
    # lay words out by hand so spans can get a background box
    xx, r = 0.0, fig.canvas.get_renderer()
    for seg in a["summary"]:
        for w in seg["text"].split(" "):
            if not w:
                continue
            t = ax.text(
                xx,
                y,
                w,
                fontsize=7.2,
                va="top",
                transform=ax.transAxes,
                bbox={"facecolor": cols[seg["err"]], "pad": 1.2, "lw": 0, "alpha": 0.8}
                if seg.get("err")
                else None,
            )
            wd = t.get_window_extent(r).width / fig.get_figwidth() / fig.dpi
            if xx + wd > 1:
                xx, y = 0.0, y - 0.1
                t.set_position((xx, y))
            xx += wd + 0.008
    for i, (k, c) in enumerate(cols.items()):
        ax.text(
            0.0 + i * 0.22,
            0.02,
            k,
            fontsize=6.5,
            transform=ax.transAxes,
            bbox={"facecolor": c, "pad": 1.2, "lw": 0, "alpha": 0.8},
        )
    fig.savefig(out / "errors.pdf", bbox_inches="tight")


def main_():
    out.mkdir(parents=True, exist_ok=True)
    for f in [rouge_cost, faith, length, errors]:
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)


if __name__ == "__main__":
    main_()
