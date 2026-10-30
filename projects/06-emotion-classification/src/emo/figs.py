from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from labcore import load  # noqa: E402

from emo import data, metrics  # noqa: E402
from emo.labels import names, nl  # noqa: E402

out = Path("report/figs")
# fixed categorical order, one colour per model family
col = {"tfidf": "#2a78d6", "bigru": "#eb6834", "student": "#1baf7a", "teacher": "#6250d6", "int8": "#e87ba4"}
plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e4e2dc", "grid.linewidth": 0.6, "axes.axisbelow": True, "savefig.dpi": 200,
    "savefig.bbox": "tight", "legend.frameon": False,
})  # fmt: skip


def per_label(m):
    ks = [("tfidf", "tfidf"), ("bigru", "bigru"), ("student-kd", "student"), ("teacher-ens", "teacher")]
    ks = [(k, c) for k, c in ks if k in m]
    ref = ks[-1][0]
    order = sorted(names, key=lambda n: m[ref]["per_label_f1"][n])
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    y = np.arange(nl)
    h = 0.8 / len(ks)
    for j, (k, c) in enumerate(ks):
        ax.barh(y + (j - (len(ks) - 1) / 2) * h, [m[k]["per_label_f1"][n] for n in order], h * 0.9,
                color=col[c], label=m[k]["label"])  # fmt: skip
    ax.set_yticks(y, order)
    ax.set_xlabel("test f1")
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.savefig(out / "per_label_f1.png")
    plt.close(fig)


def reliability(z, ge, t):
    yt = ge["test"][1]
    fig, axs = plt.subplots(1, 2, figsize=(7, 3.2), sharey=True)
    for ax, tt, ttl in [(axs[0], 1.0, "raw"), (axs[1], t, f"temperature scaled (t={t:.2f})")]:
        p = metrics.sig(z["test"] / tt)
        conf, acc, n = metrics.rel_bins(p, yt)
        m = n > 50
        ax.plot([0, 1], [0, 1], color="#9a958a", lw=1, ls="--")
        ax.plot(conf[m], acc[m], marker="o", ms=4, lw=2, color=col["student"])
        ax.set_title(f"{ttl}, ece {metrics.ece(p, yt):.4f}", fontsize=9)
        ax.set_xlabel("predicted probability")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("fraction positive")
    fig.savefig(out / "reliability.png")
    plt.close(fig)


def cooc(z, ge, th, t):
    yt = ge["test"][1]
    pred = metrics.predict(metrics.sig(z["test"] / t), np.array(th))
    # rows: gold label, cols: how often each label is predicted on those examples
    conf = (yt.T @ pred) / np.maximum(yt.sum(0)[:, None], 1)
    co = (yt.T @ yt) / np.maximum(yt.sum(0)[:, None], 1)
    np.fill_diagonal(co, np.nan)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, mat, ttl in [
        (axs[0], co, "gold co-occurrence  p(col | row)"),
        (axs[1], conf, "student: p(pred col | gold row)"),
    ]:
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(nl), names, rotation=90, fontsize=6.5)
        ax.set_yticks(range(nl), names, fontsize=6.5)
        ax.grid(False)
        ax.set_title(ttl, fontsize=9)
    fig.colorbar(im, ax=axs, shrink=0.7)
    fig.savefig(out / "cooc.png")
    plt.close(fig)


def tradeoff(m, b):
    pts = [
        ("roberta-large", "teacher", "teacher-bce", "teacher"),
        ("distilroberta + kd", "student", "student-kd", "student"),
        ("int8 onnx", "onnx-int8", "student-int8", "int8"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3), sharey=True)
    for lbl, bk, mk, c in pts:
        f, (lo, hi) = m[mk]["macro_f1"], m[mk]["macro_f1_ci"]
        for ax, x in [(axs[0], b[bk]["mb"]), (axs[1], b[bk]["ms"])]:
            ax.errorbar(x, f, yerr=[[f - lo], [hi - f]], fmt="o", ms=6, color=col[c], capsize=3, zorder=3)
            ax.annotate(lbl, (x, f), textcoords="offset points", xytext=(7, -3), ha="left", fontsize=7.5)
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlim(ax.get_xlim()[0] / 1.6, ax.get_xlim()[1] * 1.6)
    axs[0].set_ylim(0.48, 0.6)
    axs[0].set_xlabel("weights on disk (mb, log)")
    axs[1].set_xlabel(f"cpu latency per sentence (ms, {b['threads']} threads, log)")
    axs[0].set_ylabel("test macro-f1 (95% bootstrap ci)")
    fig.savefig(out / "tradeoff.png")
    plt.close(fig)


def curves(runs):
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    for k, c in [("teacher-bce", "teacher"), ("teacher-asl", "teacher"), ("student-kd", "student"),
                 ("student-hard", "student"), ("bigru", "bigru")]:  # fmt: skip
        p = Path(runs) / k / "log.json"
        if not p.exists():
            continue
        e = load(p)["ep"]
        ax.plot([d["ep"] + 1 for d in e], [d["val_macro_f1"] for d in e], marker="o", ms=3, color=col[c],
                ls="--" if k in ("teacher-asl", "student-hard") else "-", label=k)  # fmt: skip
    ax.set_xlabel("epoch")
    ax.set_ylabel("val macro-f1")
    ax.legend(fontsize=7)
    fig.savefig(out / "curves.png")
    plt.close(fig)


def main():
    out.mkdir(parents=True, exist_ok=True)
    m = load("results/metrics.json")
    ge = data.load_ge()
    z = dict(np.load("runs/student-kd/logits.npz"))
    per_label(m)
    reliability(z, ge, m["student-kd"]["temp"])
    cooc(z, ge, m["student-kd"]["th"], m["student-kd"]["temp"])
    if Path("results/bench.json").exists():
        tradeoff(m, load("results/bench.json"))
    curves("runs")


if __name__ == "__main__":
    main()
