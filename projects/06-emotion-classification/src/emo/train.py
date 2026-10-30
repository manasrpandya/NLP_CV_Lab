import argparse
import copy
import re
from functools import partial
from pathlib import Path

import numpy as np
import torch
from labcore import dev, dump, seed_all, tick
from tqdm import tqdm

from emo import data, losses, metrics
from emo.labels import names, nl


def build(mdl_id):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(mdl_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        mdl_id,
        num_labels=nl,
        id2label=dict(enumerate(names)),
        label2id={n: i for i, n in enumerate(names)},
        problem_type="multi_label_classification",
    )
    return tok, mdl


def groups(mdl, lr, decay, wd=0.01):
    # layer-wise lr decay: head gets lr, layer i gets lr * decay^(L - i), embeddings one step lower
    nlay = mdl.config.num_hidden_layers
    gs = {}
    for n, p in mdl.named_parameters():
        m = re.search(r"layer\.(\d+)\.", n)
        if m:
            depth = nlay - int(m.group(1))
        elif "embeddings" in n:
            depth = nlay + 1
        else:
            depth = 0
        nd = p.ndim == 1 or "LayerNorm" in n
        k = (depth, nd)
        if k not in gs:
            gs[k] = {"params": [], "lr": lr * decay**depth, "weight_decay": 0.0 if nd else wd}
        gs[k]["params"].append(p)
    return list(gs.values())


@torch.no_grad()
def logits(mdl, dl, d):
    mdl.eval()
    out = []
    for b in dl:
        with torch.autocast(d.type, dtype=torch.float16, enabled=d.type == "cuda"):
            z = mdl(input_ids=b["input_ids"].to(d), attention_mask=b["attention_mask"].to(d)).logits
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def val_f1(z, y):
    p = metrics.sig(z)
    th = metrics.tune_th(y, p)
    return float(metrics.f1s(y, metrics.predict(p, th)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdl", default="distilroberta-base")
    ap.add_argument("--name", required=True)
    ap.add_argument("--loss", default="bce", choices=["bce", "asl"])
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--llrd", type=float, default=0.9)
    ap.add_argument("--ep", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--warm", type=float, default=0.06)
    ap.add_argument("--kd", nargs="*", default=[], help="npz files with teacher train logits (averaged)")
    ap.add_argument("--alpha", type=float, default=0.5, help="weight of the hard-label loss when distilling")
    ap.add_argument("--kd-t", type=float, default=2.0)
    ap.add_argument("--train-logits", action="store_true", help="also save logits on the train set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--push", default=None, help="hf model repo for the weights")
    ap.add_argument("--runs-repo", default=None, help="hf dataset repo for logits / logs")
    a = ap.parse_args()

    seed_all(a.seed)
    d = dev()
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    tok, mdl = build(a.mdl)
    mdl.to(d)

    ge = data.load_ge()
    xd, _ = data.load_dair("test")
    soft = None
    if a.kd:
        soft = np.mean([np.load(f)["train"] for f in a.kd], 0).astype(np.float32)
        assert soft.shape == ge["train"][1].shape
    col = partial(data.collate, pad_id=tok.pad_token_id)
    tr = data.TextDs(*ge["train"], tok, a.max_len, soft=soft)
    dl = torch.utils.data.DataLoader(tr, a.bs, shuffle=True, collate_fn=col, num_workers=2, drop_last=True)

    def mk(x, y):
        return torch.utils.data.DataLoader(data.TextDs(x, y, tok, a.max_len), 128, collate_fn=col)

    dl_va = mk(*ge["validation"])
    yva = ge["validation"][1]

    opt = torch.optim.AdamW(groups(mdl, a.lr, a.llrd), lr=a.lr, betas=(0.9, 0.98), eps=1e-6)
    tot = a.ep * len(dl)
    nw = int(a.warm * tot)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(nw, 1) if s < nw else max(0.0, (tot - s) / max(tot - nw, 1))
    )
    scaler = torch.amp.GradScaler(enabled=d.type == "cuda")
    lossf = losses.asl if a.loss == "asl" else losses.bce

    log = {"args": vars(a), "ep": []}
    best, best_sd = -1.0, None
    with tick(log, "train_sec"):
        for ep in range(a.ep):
            mdl.train()
            tl = 0.0
            for b in tqdm(dl, desc=f"ep{ep}", mininterval=30):
                with torch.autocast(d.type, dtype=torch.float16, enabled=d.type == "cuda"):
                    z = mdl(input_ids=b["input_ids"].to(d), attention_mask=b["attention_mask"].to(d)).logits
                y = b["y"].to(d)
                los = lossf(z, y)
                if soft is not None:
                    los = a.alpha * los + (1 - a.alpha) * losses.kd(z, b["soft"].to(d), a.kd_t)
                opt.zero_grad(set_to_none=True)
                scaler.scale(los).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sch.step()
                tl += los.item()
            f = val_f1(logits(mdl, dl_va, d), yva)
            log["ep"].append({"ep": ep, "loss": tl / len(dl), "val_macro_f1": f})
            print(f"ep {ep} loss {tl / len(dl):.4f} val macro-f1 {f:.4f}", flush=True)
            if f > best:
                best, best_sd = f, copy.deepcopy({k: v.cpu() for k, v in mdl.state_dict().items()})

    mdl.load_state_dict(best_sd)
    log["best_val_macro_f1"] = best
    zs = {
        "val": logits(mdl, dl_va, d),
        "test": logits(mdl, mk(*ge["test"]), d),
        "dair": logits(mdl, mk(xd, np.zeros((len(xd), nl), np.float32)), d),
    }
    if a.train_logits:
        zs["train"] = logits(mdl, mk(*ge["train"]), d)
    np.savez_compressed(out / "logits.npz", **zs)
    dump(log, out / "log.json")
    mdl.save_pretrained(out / "model")
    tok.save_pretrained(out / "model")

    if a.push or a.runs_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        if a.push:
            api.create_repo(a.push, exist_ok=True)
            api.upload_folder(repo_id=a.push, folder_path=out / "model", commit_message=f"{a.name} weights")
        if a.runs_repo:
            api.create_repo(a.runs_repo, repo_type="dataset", exist_ok=True)
            for f in ["logits.npz", "log.json"]:
                api.upload_file(
                    path_or_fileobj=out / f, path_in_repo=f"{a.name}/{f}", repo_id=a.runs_repo, repo_type="dataset"
                )
    print("done", a.name, best)


if __name__ == "__main__":
    main()
