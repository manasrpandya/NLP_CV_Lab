import numpy as np

# goemotions "simplified" label order (index = class id in the hf dataset)
names = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
]  # fmt: skip
nl = len(names)
idx = {n: i for i, n in enumerate(names)}

# ekman grouping from the goemotions repo (ekman_mapping.json) plus neutral
ekman = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": [
        "joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism",
        "relief", "pride", "admiration", "desire", "caring",
    ],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}  # fmt: skip
ek_names = list(ekman)
ne = len(ek_names)

# fine -> ekman group id
fine2ek = np.zeros(nl, dtype=np.int64)
for j, g in enumerate(ek_names):
    for n in ekman[g]:
        fine2ek[idx[n]] = j

# dair-ai/emotion (twitter, single label) -> ekman. love folds into joy.
dair_names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
dair2ek = np.array([ek_names.index({"love": "joy"}.get(n, n)) for n in dair_names])


def multihot(lbls, n=nl):
    y = np.zeros((len(lbls), n), dtype=np.float32)
    for i, ls in enumerate(lbls):
        y[i, list(ls)] = 1.0
    return y


def to_ekman(y):
    # fine multi-hot (n, 28) -> ekman multi-hot (n, 7): a group is on if any member is on
    out = np.zeros((y.shape[0], ne), dtype=y.dtype)
    for j in range(ne):
        out[:, j] = y[:, fine2ek == j].max(1)
    return out
