import numpy as np

cnn_id = "abisee/cnn_dailymail"
xsum_id = "EdinburghNLP/xsum"
sets = {
    "cnndm": {"id": cnn_id, "cfg": "3.0.0", "src": "article", "tgt": "highlights"},
    "xsum": {"id": xsum_id, "cfg": None, "src": "document", "tgt": "summary"},
}
buckets = ["short", "medium", "long"]


def load(name, split):
    from datasets import load_dataset

    s = sets[name]
    ds = load_dataset(s["id"], s["cfg"])[split] if s["cfg"] else load_dataset(s["id"])[split]
    return ds, s["src"], s["tgt"]


def subset(n_total, n, seed=0):
    # fixed random subset, same for every model
    return np.sort(np.random.RandomState(seed).permutation(n_total)[:n])


def cuts(refs):
    # tertiles of the reference length in words
    n = [len(r.split()) for r in refs]
    return [float(c) for c in np.quantile(n, [1 / 3, 2 / 3])]


def bucket(nw, cs):
    return buckets[0] if nw <= cs[0] else buckets[1] if nw <= cs[1] else buckets[2]


def prefix(b):
    return f"summarize {b}: "


def with_prefix(src, b):
    return prefix(b) + src
