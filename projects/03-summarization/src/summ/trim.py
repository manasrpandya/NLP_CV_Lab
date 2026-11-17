import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def count(tok, texts):
    c = Counter()
    for i in range(0, len(texts), 2000):
        for ids in tok(texts[i : i + 2000], add_special_tokens=False)["input_ids"]:
            c.update(ids)
    return c


def pick(tj, cnt, k):
    # keep specials, the single-character (byte level) tokens, then the most frequent tokens up to k.
    # every kept token also needs the pieces of the merges that build it so that bpe can still reach it.
    vocab = tj["model"]["vocab"]
    inv = {i: t for t, i in vocab.items()}
    mer = [tuple(m) if isinstance(m, list) else tuple(m.split(" ")) for m in tj["model"]["merges"]]
    prod = {}
    for a, b in mer:
        prod.setdefault(a + b, []).append((a, b))
    keep = set(range(4)) | {i for t, i in vocab.items() if len(t) == 1}

    def add(t):
        i = vocab[t]
        if i in keep and t in done:
            return
        keep.add(i)
        done.add(t)
        # every merge that can build t, since bpe may reach it through any of them depending on the word
        for pair in prod.get(t, []):
            for p in pair:
                add(p)

    done = set()
    for i in list(keep):
        done.add(inv[i])
    for i, _ in cnt.most_common():
        if len(keep) >= k:
            break
        add(inv[i])
    # the last id is <mask> in bart, keep it as well
    keep.add(max(vocab.values()))
    return sorted(keep), mer


def trim_tokenizer(tj, keep, mer):
    inv = {i: t for t, i in tj["model"]["vocab"].items()}
    new = {inv[o]: n for n, o in enumerate(keep)}
    out = json.loads(json.dumps(tj))
    out["model"]["vocab"] = new
    ms = [m for m in mer if m[0] in new and m[1] in new and m[0] + m[1] in new]
    out["model"]["merges"] = [list(m) for m in ms]
    for t in out["added_tokens"]:
        t["id"] = new[t["content"]]
    return out


def trim_model(mdl, keep):
    import torch

    k = torch.tensor(keep)
    w = mdl.get_input_embeddings().weight.data[k].clone()
    b = mdl.final_logits_bias[:, k].clone() if hasattr(mdl, "final_logits_bias") else None
    mdl.resize_token_embeddings(len(keep))
    mdl.get_input_embeddings().weight.data.copy_(w)
    if b is not None:
        mdl.register_buffer("final_logits_bias", b)
    mdl.config.vocab_size = len(keep)
    return mdl


def tidy(out, n):
    # the old vocab files and the <mask> id in the tokenizer config no longer match the trimmed vocab
    out = Path(out)
    for f in ["vocab.json", "merges.txt"]:
        (out / f).unlink(missing_ok=True)
    tc = json.loads((out / "tokenizer_config.json").read_text())
    tc["added_tokens_decoder"] = {
        (str(n - 1) if v["content"] == "<mask>" else k): v for k, v in tc["added_tokens_decoder"].items()
    }
    (out / "tokenizer_config.json").write_text(json.dumps(tc, indent=2))
    gc = json.loads((out / "generation_config.json").read_text())
    gc.update({"num_beams": 1, "no_repeat_ngram_size": 3, "max_length": 142, "early_stopping": False})
    (out / "generation_config.json").write_text(json.dumps(gc, indent=2))


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    from summ import data
    from summ.train import order

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="manasrpandya/cnndm-summ")
    ap.add_argument("--out", default="runs/trimmed")
    ap.add_argument("--k", type=int, default=20000)
    ap.add_argument("--n", type=int, default=40000)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.src)
    ds, sc, tc = data.load("cnndm", "train")
    sub = ds.select([int(i) for i in order(len(ds))[: a.n]])
    cnt = count(tok, list(sub[sc]) + list(sub[tc]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out)
    tj = json.loads((out / "tokenizer.json").read_text())
    keep, mer = pick(tj, cnt, a.k)
    (out / "tokenizer.json").write_text(json.dumps(trim_tokenizer(tj, keep, mer)))
    np.save(out / "keep.npy", np.array(keep))
    mdl = AutoModelForSeq2SeqLM.from_pretrained(a.src)
    trim_model(mdl, keep).save_pretrained(out)
    tidy(out, len(keep))
    print("kept", len(keep), "of", len(tok), "token types seen", len(cnt))


if __name__ == "__main__":
    main()
