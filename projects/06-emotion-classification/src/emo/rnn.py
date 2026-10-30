import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from labcore import dev, dump, seed_all, tick

from emo import data, metrics
from emo.labels import nl

glove_id = "sentence-transformers/average_word_embeddings_glove.6B.300d"
pat = re.compile(r"\[name\]|\[religion\]|[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def toks(s):
    return pat.findall(s.lower())


def load_glove():
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file

    voc = json.load(open(hf_hub_download(glove_id, "0_WordEmbeddings/whitespacetokenizer_config.json")))[
        "vocab"
    ]
    w = load_file(hf_hub_download(glove_id, "0_WordEmbeddings/model.safetensors"))
    return voc, next(iter(w.values()))


def build_vocab(xs, gvoc, gw, min_n=2):
    # train words that glove knows get its vector, frequent unknown ones get a random init
    cnt = Counter(t for s in xs for t in toks(s))
    gi = {w: i for i, w in enumerate(gvoc)}
    itos = ["<pad>", "<unk>"] + [w for w, c in cnt.most_common() if w in gi or c >= min_n]
    rng = np.random.default_rng(0)
    emb = rng.normal(0, 0.3, (len(itos), gw.shape[1])).astype(np.float32)
    emb[0] = 0
    hit = 0
    for i, w in enumerate(itos[2:], 2):
        if w in gi:
            emb[i] = gw[gi[w]]
            hit += 1
    return {w: i for i, w in enumerate(itos)}, emb, hit


def encode(xs, stoi, max_len=64):
    return [[stoi.get(t, 1) for t in toks(s)][:max_len] or [1] for s in xs]


class BiGRU(nn.Module):
    def __init__(self, emb, h=256, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb), freeze=False, padding_idx=0)
        self.drop = nn.Dropout(drop)
        self.rnn = nn.GRU(emb.shape[1], h, batch_first=True, bidirectional=True)
        self.head = nn.Linear(4 * h, nl)

    def forward(self, ids, mask):
        h, _ = self.rnn(self.drop(self.emb(ids)))
        m = mask.unsqueeze(-1).float()
        mx = h.masked_fill(m == 0, -1e4).max(1).values
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1)
        return self.head(self.drop(torch.cat([mx, mean], -1)))


def batches(enc, y, bs, shuffle=False, rng=None):
    order = rng.permutation(len(enc)) if shuffle else np.arange(len(enc))
    for i in range(0, len(enc), bs):
        j = order[i : i + bs]
        b = data.collate([{"ids": enc[k], "y": y[k]} for k in j], pad_id=0)
        yield b["input_ids"], b["attention_mask"], b["y"]


@torch.no_grad()
def logits(mdl, enc, d):
    mdl.eval()
    y0 = np.zeros((len(enc), nl), np.float32)
    return np.concatenate([mdl(i.to(d), m.to(d)).cpu().numpy() for i, m, _ in batches(enc, y0, 256)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--name", default="bigru")
    ap.add_argument("--runs-repo", default=None)
    a = ap.parse_args()
    seed_all(0)
    d = dev()
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)

    ge = data.load_ge()
    xd, _ = data.load_dair("test")
    stoi, emb, hit = build_vocab(ge["train"][0], *load_glove())
    print(f"vocab {len(stoi)}, glove hits {hit}")
    enc = {s: encode(ge[s][0], stoi) for s in ge}
    mdl = BiGRU(emb).to(d)
    # embeddings get a smaller lr so the glove geometry isn't wrecked early on
    opt = torch.optim.Adam(
        [
            {"params": mdl.emb.parameters(), "lr": a.lr / 10},
            {"params": [p for n, p in mdl.named_parameters() if not n.startswith("emb")]},
        ],
        lr=a.lr,
    )
    lossf = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(0)
    log = {"args": vars(a), "vocab": len(stoi), "glove_hits": hit, "ep": []}
    best, best_sd = -1.0, None
    with tick(log, "train_sec"):
        for ep in range(a.ep):
            mdl.train()
            tl, nb = 0.0, 0
            for i, m, y in batches(enc["train"], ge["train"][1], a.bs, True, rng):
                los = lossf(mdl(i.to(d), m.to(d)), y.to(d))
                opt.zero_grad()
                los.backward()
                nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step()
                tl, nb = tl + los.item(), nb + 1
            zv = logits(mdl, enc["validation"], d)
            p = metrics.sig(zv)
            f = float(
                metrics.f1s(
                    ge["validation"][1], metrics.predict(p, metrics.tune_th(ge["validation"][1], p))
                ).mean()
            )
            log["ep"].append({"ep": ep, "loss": tl / nb, "val_macro_f1": f})
            print(f"ep {ep} loss {tl / nb:.4f} val macro-f1 {f:.4f}", flush=True)
            if f > best:
                best, best_sd = f, {k: v.detach().clone() for k, v in mdl.state_dict().items()}
    mdl.load_state_dict(best_sd)
    log["best_val_macro_f1"] = best
    zs = {"val": logits(mdl, enc["validation"], d), "test": logits(mdl, enc["test"], d)}
    zs["dair"] = logits(mdl, encode(xd, stoi), d)
    np.savez_compressed(out / "logits.npz", **zs)
    dump(log, out / "log.json")
    torch.save(mdl.state_dict(), out / "bigru.pt")
    if a.runs_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(a.runs_repo, repo_type="dataset", exist_ok=True)
        for f in ["logits.npz", "log.json"]:
            api.upload_file(
                path_or_fileobj=out / f,
                path_in_repo=f"{a.name}/{f}",
                repo_id=a.runs_repo,
                repo_type="dataset",
            )


if __name__ == "__main__":
    main()
