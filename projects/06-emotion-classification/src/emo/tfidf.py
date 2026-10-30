import argparse
from pathlib import Path

import numpy as np
from labcore import dump, tick
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from emo import data, metrics


def feats(xtr):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000)
    c = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=200_000
    )
    w.fit(xtr)
    c.fit(xtr)
    return lambda xs: hstack([w.transform(xs), c.transform(xs)]).tocsr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs")
    ap.add_argument("--cs", type=float, nargs="+", default=[1.0, 4.0, 16.0])
    a = ap.parse_args()
    out = Path(a.out) / "tfidf"
    ge = data.load_ge()
    xd, _ = data.load_dair("test")
    f = feats(ge["train"][0])
    xs = {k: f(ge[k][0]) for k in ge}
    log = {"c": {}}
    best = (-1, None, None)
    for c in a.cs:
        with tick(log, f"sec_c{c}"):
            clf = OneVsRestClassifier(
                LogisticRegression(C=c, max_iter=2000, class_weight="balanced"), n_jobs=4
            )
            clf.fit(xs["train"], ge["train"][1])
        zv = clf.decision_function(xs["validation"])
        yv = ge["validation"][1]
        pv = metrics.sig(zv)
        mf = float(metrics.f1s(yv, metrics.predict(pv, metrics.tune_th(yv, pv))).mean())
        log["c"][c] = mf
        print(f"c={c} val macro-f1 {mf:.4f}", flush=True)
        if mf > best[0]:
            best = (mf, c, clf)
    clf = best[2]
    log["best_c"] = best[1]
    zs = {"val": clf.decision_function(xs["validation"]), "test": clf.decision_function(xs["test"])}
    zs["dair"] = clf.decision_function(f(xd))
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "logits.npz", **{k: v.astype(np.float32) for k, v in zs.items()})
    dump(log, out / "log.json")


if __name__ == "__main__":
    main()
