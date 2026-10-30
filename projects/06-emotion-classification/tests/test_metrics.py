import numpy as np
from emo import metrics


def test_f1s_and_micro():
    y = np.array([[1, 0], [1, 1], [0, 0]], np.float32)
    p = np.array([[1, 0], [0, 1], [1, 0]], np.float32)
    f = metrics.f1s(y, p)
    assert np.allclose(f, [0.5, 1.0])
    assert np.isclose(metrics.micro(y, p), 2 * 2 / (2 * 2 + 1 + 1))


def test_tune_th_separable():
    rng = np.random.default_rng(0)
    y = (rng.random((500, 3)) < 0.2).astype(np.float32)
    # positives sit above 0.6, negatives below 0.4, so a threshold in between gets f1 = 1
    p = np.where(y == 1, rng.uniform(0.6, 1, y.shape), rng.uniform(0, 0.4, y.shape))
    th = metrics.tune_th(y, p)
    assert ((th >= 0.4) & (th <= 0.6)).all()
    assert np.allclose(metrics.f1s(y, metrics.predict(p, th, top1=False)), 1)


def test_predict_top1_fallback():
    p = np.array([[0.1, 0.3], [0.9, 0.2]])
    th = np.array([0.5, 0.4])
    assert metrics.predict(p, th, top1=False).tolist() == [[0, 0], [1, 0]]
    # first row: 0.3/0.4 beats 0.1/0.5
    assert metrics.predict(p, th).tolist() == [[0, 1], [1, 0]]


def test_fit_temp_recovers_scale():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, (20000, 4))
    y = (rng.random(z.shape) < metrics.sig(z)).astype(np.float32)
    # logits that are 2x too sharp should get t close to 2
    assert abs(metrics.fit_temp(2 * z, y) - 2) < 0.15


def test_ece():
    rng = np.random.default_rng(2)
    p = rng.random((20000, 2))
    y = (rng.random(p.shape) < p).astype(np.float32)
    assert metrics.ece(p, y) < 0.02
    assert metrics.ece(np.ones_like(p) * 0.9, np.zeros_like(p)) > 0.85
