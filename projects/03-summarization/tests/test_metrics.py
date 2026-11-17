import numpy as np

from summ import data
from summ.metrics import boot, consistency, novel, rouge


def test_rouge_identical_and_disjoint():
    r = rouge(["the cat sat on the mat. it was happy."], ["the cat sat on the mat. it was happy."])
    assert np.allclose(r, 100)
    r = rouge(["the cat sat on the mat."], ["quantum chromodynamics explains hadrons."])
    assert np.allclose(r, 0)


def test_rouge_stemming():
    r = rouge(["the cats are running"], ["the cat is run"])
    assert r[0, 0] > 60


def test_novel_bigrams():
    src = "the quick brown fox jumps over the lazy dog"
    assert novel(src, "the quick brown fox") == 0.0
    assert novel(src, "a slow red cat") == 1.0
    assert abs(novel(src, "the quick brown cat") - 1 / 3) < 1e-9
    assert novel(src, "") == 0.0


def test_consistency_max_then_mean():
    # two summary sentences, three source sentences
    m = np.array([[0.1, 0.9, 0.2], [0.3, 0.2, 0.5]])
    assert np.allclose(consistency([m]), [0.7])
    assert consistency([np.zeros((0, 0))]) == [0.0]


def test_boot_ci_contains_mean():
    x = np.random.RandomState(0).rand(200)
    lo, hi = boot(x, 200)
    assert lo < x.mean() < hi


def test_length_buckets_and_prefix():
    refs = ["a " * n for n in range(1, 31)]
    cs = data.cuts(refs)
    assert cs[0] < cs[1]
    assert data.bucket(1, cs) == "short"
    assert data.bucket(15, cs) == "medium"
    assert data.bucket(30, cs) == "long"
    assert data.with_prefix("text", "short") == "summarize short: text"


def test_subset_is_fixed_and_sorted():
    a, b = data.subset(1000, 50), data.subset(1000, 50)
    assert (a == b).all() and (np.diff(a) > 0).all() and len(a) == 50
