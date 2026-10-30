import argparse
from pathlib import Path

import numpy as np
from labcore import dump
from sklearn.metrics import accuracy_score, f1_score

from emo import data, metrics
from emo.labels import dair2ek, ek_names, names, to_ekman

# run name -> row label in the tables
rows = {
    "tfidf": "tf-idf + logreg",
    "bigru": "bigru + glove",
    "student-hard": "distilroberta (hard labels)",
    "student-kd": "distilroberta + kd",
    "teacher-bce": "roberta-large (bce)",
    "teacher-asl": "roberta-large (asl)",
    "teacher-ens": "roberta-large ens.",
}
ood_cls = [ek_names.index(n) for n in ["anger", "fear", "joy", "sadness", "surprise"]]


def score(z, ge, yd):
    yv, yt = ge["validation"][1], ge["test"][1]
    t = metrics.fit_temp(z["val"], yv)
    p0 = metrics.sig(z["test"])
    pv, pt = metrics.sig(z["val"] / t), metrics.sig(z["test"] / t)
    th = metrics.tune_th(yv, pv)
    pred = metrics.predict(pt, th)
    f = metrics.f1s(yt, pred)
    # ekman view: a group's prob is the max over its members, thresholds re-tuned on val
    ev, et = to_ekman(pv), to_ekman(pt)
    eth = metrics.tune_th(to_ekman(yv), ev)
    ef = metrics.f1s(to_ekman(yt), metrics.predict(et, eth))
    # zero-shot on dair-ai/emotion: argmax over the five ekman groups it has
    pd = to_ekman(metrics.sig(z["dair"] / t))[:, ood_cls]
    yh = np.array(ood_cls)[pd.argmax(1)]
    yg = dair2ek[yd]
    return {
        "temp": t,
        "th": th.tolist(),
        "macro_f1": float(f.mean()),
        "micro_f1": metrics.micro(yt, pred),
        "macro_f1_at_05": float(metrics.f1s(yt, metrics.predict(pt, 0.5, top1=False)).mean()),
        "per_label_f1": dict(zip(names, f.round(4).tolist())),
        "ece_raw": metrics.ece(p0, yt),
        "ece_cal": metrics.ece(pt, yt),
        "ece_pos_raw": metrics.ece_pos(p0, yt),
        "ece_pos_cal": metrics.ece_pos(pt, yt),
        "ekman_macro_f1": float(ef.mean()),
        "ekman_f1": dict(zip(ek_names, ef.round(4).tolist())),
        "ood_acc": float(accuracy_score(yg, yh)),
        "ood_macro_f1": float(f1_score(yg, yh, average="macro")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="results/metrics.json")
    a = ap.parse_args()
    ge = data.load_ge()
    _, yd = data.load_dair("test")
    zs = {}
    for k in rows:
        p = Path(a.runs) / k / "logits.npz"
        if p.exists():
            zs[k] = dict(np.load(p))
    if "teacher-bce" in zs and "teacher-asl" in zs:
        zs["teacher-ens"] = {
            s: (zs["teacher-bce"][s] + zs["teacher-asl"][s]) / 2 for s in ["val", "test", "dair"]
        }
    res = {}
    for k, z in zs.items():
        res[k] = {"label": rows[k], **score(z, ge, yd)}
        r = res[k]
        print(
            f"{k:14s} macro {r['macro_f1']:.4f} micro {r['micro_f1']:.4f} ekman {r['ekman_macro_f1']:.4f} "
            f"ood acc {r['ood_acc']:.4f} ece {r['ece_raw']:.4f}->{r['ece_cal']:.4f} t={r['temp']:.2f}"
        )
    dump(res, a.out)


if __name__ == "__main__":
    main()
