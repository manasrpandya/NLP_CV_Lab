import numpy as np
import torch

from summ.metrics import consistency
from summ.text import sents

nli_id = "cross-encoder/nli-deberta-v3-xsmall"


def load(d="cpu"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(nli_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(nli_id).eval().to(d)
    ent = [i for i, n in mdl.config.id2label.items() if n.lower() == "entailment"][0]
    return tok, mdl, ent


@torch.no_grad()
def entail(tok, mdl, ent, prem, hyp, bs=64, max_len=256):
    d = next(mdl.parameters()).device
    out = []
    for i in range(0, len(prem), bs):
        enc = tok(
            prem[i : i + bs],
            hyp[i : i + bs],
            truncation=True,
            max_length=max_len,
            padding=True,
            return_tensors="pt",
        ).to(d)
        out.append(mdl(**enc).logits.softmax(-1)[:, ent].float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def matrices(tok, mdl, ent, srcs, hyps, bs=64):
    # one (summary sentence x source sentence) entailment matrix per example
    pairs, idx = [], []
    for i, (s, h) in enumerate(zip(srcs, hyps)):
        ss, hs = sents(s), sents(h)
        for a in range(len(hs)):
            for b in range(len(ss)):
                pairs.append((ss[b], hs[a]))
                idx.append((i, a, b, len(hs), len(ss)))
    p = entail(tok, mdl, ent, [x[0] for x in pairs], [x[1] for x in pairs], bs)
    mats = [None] * len(srcs)
    for (i, a, b, nh, ns), v in zip(idx, p):
        if mats[i] is None:
            mats[i] = np.zeros((nh, ns))
        mats[i][a, b] = v
    return [m if m is not None else np.zeros((0, 0)) for m in mats]


def score(tok, mdl, ent, srcs, hyps, bs=64):
    return consistency(matrices(tok, mdl, ent, srcs, hyps, bs))
