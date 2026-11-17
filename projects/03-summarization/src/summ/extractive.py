import numpy as np

from summ.text import sents


def lead(x, k=3):
    return " ".join(sents(x)[:k])


def textrank(x, k=3, d=0.85, it=50):
    # sentences as nodes, tf-idf cosine as edge weight, plain power iteration. summary keeps source order.
    from sklearn.feature_extraction.text import TfidfVectorizer

    ss = sents(x)
    if len(ss) <= k:
        return " ".join(ss)
    try:
        m = TfidfVectorizer(stop_words="english").fit_transform(ss)
    except ValueError:
        return " ".join(ss[:k])
    w = (m @ m.T).toarray()
    np.fill_diagonal(w, 0)
    rs = w.sum(1, keepdims=True)
    p = np.divide(w, rs, out=np.full_like(w, 1 / len(ss)), where=rs > 0)
    r = np.full(len(ss), 1 / len(ss))
    for _ in range(it):
        r2 = (1 - d) / len(ss) + d * p.T @ r
        if np.abs(r2 - r).sum() < 1e-6:
            r = r2
            break
        r = r2
    top = sorted(np.argsort(-r, kind="stable")[:k])
    return " ".join(ss[i] for i in top)
