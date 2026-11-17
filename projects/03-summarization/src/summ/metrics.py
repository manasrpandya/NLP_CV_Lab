import numpy as np

from summ.text import sents, words


def rouge(refs, hyps):
    # rouge-1/2/lsum f1 with stemming, lsum sees one sentence per line
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeLsum"], use_stemmer=True)
    rows = []
    for r, h in zip(refs, hyps):
        s = sc.score("\n".join(sents(r)), "\n".join(sents(h)))
        rows.append([s["rouge1"].fmeasure, s["rouge2"].fmeasure, s["rougeLsum"].fmeasure])
    return np.array(rows) * 100


def ngrams(w, n):
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def novel(src, hyp, n=2):
    # share of summary n-grams that never occur in the source
    h = ngrams(words(hyp), n)
    if not h:
        return 0.0
    return 1 - len(h & ngrams(words(src), n)) / len(h)


def consistency(mats):
    # summac-zs style: for each summary sentence take its best entailment over the source sentences,
    # then average over summary sentences. mats[i] has shape (n_summary_sents, n_source_sents).
    out = []
    for m in mats:
        m = np.asarray(m)
        out.append(float(m.max(1).mean()) if m.size else 0.0)
    return out


def boot(x, n=1000, seed=0):
    # 95% ci of the mean over examples
    x = np.asarray(x)
    r = np.random.RandomState(seed)
    ms = [x[r.randint(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
