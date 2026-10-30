import numpy as np
from emo.labels import dair2ek, dair_names, ek_names, ekman, fine2ek, idx, multihot, names, nl, to_ekman


def test_ekman_covers_every_label_once():
    flat = [n for g in ekman.values() for n in g]
    assert sorted(flat) == sorted(names)
    assert len(flat) == nl == 28


def test_fine2ek():
    assert ek_names[fine2ek[idx["annoyance"]]] == "anger"
    assert ek_names[fine2ek[idx["curiosity"]]] == "surprise"
    assert ek_names[fine2ek[idx["neutral"]]] == "neutral"


def test_to_ekman_any():
    y = multihot([[idx["grief"]], [idx["joy"], idx["fear"]], []])
    e = to_ekman(y)
    assert e.shape == (3, 7)
    assert e[0].tolist() == [0, 0, 0, 0, 1, 0, 0]
    assert e[1, ek_names.index("joy")] == 1 and e[1, ek_names.index("fear")] == 1
    assert e[2].sum() == 0


def test_to_ekman_probs_is_max():
    p = np.zeros((1, nl), np.float32)
    p[0, idx["anger"]], p[0, idx["annoyance"]] = 0.2, 0.7
    assert np.isclose(to_ekman(p)[0, ek_names.index("anger")], 0.7)


def test_dair_mapping():
    got = {d: ek_names[e] for d, e in zip(dair_names, dair2ek)}
    assert got == {
        "sadness": "sadness",
        "joy": "joy",
        "love": "joy",
        "anger": "anger",
        "fear": "fear",
        "surprise": "surprise",
    }
