import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from labcore import dev, dump, seed_all, tick
from tqdm import tqdm

from summ import data
from summ.metrics import rouge


def order(n_total, seed=1):
    return np.random.RandomState(seed).permutation(n_total)


class Ds(torch.utils.data.Dataset):
    def __init__(self, src, tgt, tok, max_src, max_tgt):
        self.x = tok(src, truncation=True, max_length=max_src)["input_ids"]
        self.y = tok(text_target=tgt, truncation=True, max_length=max_tgt)["input_ids"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def collate(b, pad):
    nx, ny = max(len(x) for x, _ in b), max(len(y) for _, y in b)
    ids = torch.full((len(b), nx), pad, dtype=torch.long)
    lbl = torch.full((len(b), ny), -100, dtype=torch.long)
    for i, (x, y) in enumerate(b):
        ids[i, : len(x)] = torch.tensor(x)
        lbl[i, : len(y)] = torch.tensor(y)
    return {"input_ids": ids, "attention_mask": (ids != pad).long(), "labels": lbl}


def rows(name, split, idx, tgts=None, cs=None, plain=False):
    # source texts (with the length prefix) and targets for the chosen rows
    ds, sc, tc = data.load(name, split)
    sub = ds.select([int(i) for i in idx])
    src, tgt = list(sub[sc]), list(sub[tc])
    if tgts is not None:
        ids = list(sub["id"])
        tgt = [tgts[i] for i in ids]
    if not plain:
        src = [data.with_prefix(s, data.bucket(len(t.split()), cs)) for s, t in zip(src, tgt)]
    return src, tgt


@torch.no_grad()
def quick_val(mdl, tok, src, tgt, d, bs=32):
    mdl.eval()
    hyp = []
    for i in range(0, len(src), bs):
        enc = tok(src[i : i + bs], truncation=True, max_length=512, padding=True, return_tensors="pt").to(d)
        with torch.autocast(d.type, dtype=torch.float16, enabled=d.type == "cuda"):
            o = mdl.generate(**enc, num_beams=2, max_length=142, no_repeat_ngram_size=3)
        hyp += tok.batch_decode(o, skip_special_tokens=True)
    return float(rouge(tgt, hyp).mean(0)[0]), hyp


def main():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--mdl", default="facebook/bart-base")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="cnndm", choices=list(data.sets))
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--targets", default=None, help="jsonl with {id, hyp} to use as targets (seq-level kd)")
    ap.add_argument("--plain", action="store_true", help="no length prefix")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max-src", type=int, default=512)
    ap.add_argument("--max-tgt", type=int, default=128)
    ap.add_argument("--warm", type=float, default=0.05)
    ap.add_argument("--ls", type=float, default=0.1)
    ap.add_argument("--nval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--push", default=None)
    ap.add_argument("--runs-repo", default=None)
    a = ap.parse_args()

    seed_all(a.seed)
    d = dev()
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.mdl)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(a.mdl).to(d)

    ds, sc, tc = data.load(a.data, "train")
    # length buckets come from the gold reference lengths of the whole train set, whatever the targets are
    cs = data.cuts(ds[tc]) if not a.plain else [0, 0]
    idx = order(len(ds))[: a.n]
    tg = None
    if a.targets:
        tg = {j["id"]: j["hyp"] for j in map(json.loads, Path(a.targets).read_text().splitlines())}
        all_ids = ds["id"]
        idx = np.array([i for i in idx if all_ids[int(i)] in tg])
    sub = ds.select([int(i) for i in idx])
    src, gold = list(sub[sc]), list(sub[tc])
    tgt = [tg[i] for i in sub["id"]] if tg else gold
    if not a.plain:
        src = [data.with_prefix(s, data.bucket(len(g.split()), cs)) for s, g in zip(src, gold)]
    print("train rows", len(src), "cuts", cs, flush=True)

    vd, _, _ = data.load(a.data, "validation")
    vi = data.subset(len(vd), a.nval, seed=2)
    vsub = vd.select([int(i) for i in vi])
    vsrc, vref = list(vsub[sc]), list(vsub[tc])
    if not a.plain:
        vsrc = [data.with_prefix(s, "medium") for s in vsrc]

    tr = Ds(src, tgt, tok, a.max_src, a.max_tgt)
    dl = torch.utils.data.DataLoader(
        tr,
        a.bs,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        num_workers=2,
        drop_last=True,
    )
    nd = ["bias", "layer_norm", "layernorm"]
    ps = [
        {
            "params": [p for n, p in mdl.named_parameters() if not any(k in n.lower() for k in nd)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in mdl.named_parameters() if any(k in n.lower() for k in nd)],
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.AdamW(ps, lr=a.lr)
    tot = a.ep * len(dl)
    nw = int(a.warm * tot)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(nw, 1) if s < nw else max(0.0, (tot - s) / max(tot - nw, 1))
    )
    scaler = torch.amp.GradScaler(enabled=d.type == "cuda")
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=a.ls)

    log = {"args": vars(a), "cuts": cs, "ep": []}
    best, best_sd = -1.0, None
    with tick(log, "train_sec"):
        for ep in range(a.ep):
            mdl.train()
            tl = 0.0
            for b in tqdm(dl, desc=f"ep{ep}", mininterval=60):
                b = {k: v.to(d) for k, v in b.items()}
                with torch.autocast(d.type, dtype=torch.float16, enabled=d.type == "cuda"):
                    z = mdl(
                        input_ids=b["input_ids"], attention_mask=b["attention_mask"], labels=b["labels"]
                    ).logits
                los = ce(z.float().view(-1, z.size(-1)), b["labels"].view(-1))
                opt.zero_grad(set_to_none=True)
                scaler.scale(los).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
                tl += los.item()
            r1, _ = quick_val(mdl, tok, vsrc, vref, d)
            log["ep"].append({"ep": ep, "loss": tl / len(dl), "val_r1": r1})
            print(f"ep {ep} loss {tl / len(dl):.4f} val rouge1 {r1:.2f}", flush=True)
            if r1 > best:
                best, best_sd = r1, copy.deepcopy({k: v.cpu() for k, v in mdl.state_dict().items()})

    mdl.load_state_dict(best_sd)
    log["best_val_r1"] = best
    dump(log, out / "log.json")
    mdl.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")
    json.dump({"cuts": cs, "buckets": data.buckets}, open(out / "model" / "buckets.json", "w"))

    if a.push or a.runs_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        if a.push:
            api.create_repo(a.push, exist_ok=True)
            api.upload_folder(repo_id=a.push, folder_path=out / "model", commit_message=f"{a.name} weights")
        if a.runs_repo:
            api.create_repo(a.runs_repo, repo_type="dataset", exist_ok=True)
            api.upload_file(
                path_or_fileobj=out / "log.json",
                path_in_repo=f"{a.name}/log.json",
                repo_id=a.runs_repo,
                repo_type="dataset",
            )
    print("done", a.name, best)


if __name__ == "__main__":
    main()
