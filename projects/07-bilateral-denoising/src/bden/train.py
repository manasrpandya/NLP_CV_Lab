import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from labcore import dev, dump, seed_all

from bden import data
from bden.metrics import psnr
from bden.model import build

here = Path(__file__).resolve().parents[2]


def crops(ims, bs, p, gen):
    # random p x p crops from a list of gpu images, with flips / 90 deg rotations
    out = []
    for i in torch.randint(len(ims), (bs,), generator=gen).tolist():
        im = ims[i]
        y = int(torch.randint(im.shape[-2] - p + 1, (1,), generator=gen))
        x = int(torch.randint(im.shape[-1] - p + 1, (1,), generator=gen))
        c = im[..., y : y + p, x : x + p]
        c = torch.rot90(c, int(torch.randint(4, (1,), generator=gen)), (-2, -1))
        out.append(c.flip(-1) if torch.rand(1, generator=gen) < 0.5 else c)
    return torch.stack(out)


@torch.no_grad()
def val(mdl, pairs, d):
    mdl.eval()
    s = [psnr(x, mdl(torch.from_numpy(y)[None, None].to(d))[0, 0].float().cpu().numpy()) for x, y in pairs]
    mdl.train()
    return float(np.mean(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mdl", default="nbf")
    ap.add_argument("--its", type=int, default=9000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--p", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--smax", type=float, default=55)
    a = ap.parse_args()

    seed_all(0)
    d = dev()
    gen = torch.Generator().manual_seed(0)
    ims = [torch.from_numpy(data.read(p))[None].to(d) for p in data.files("train")]
    vpairs = data.noisy("val", 25, seed=1)[:16]
    mdl = build(a.mdl).to(d)
    n_par = sum(p.numel() for p in mdl.parameters())
    print(a.mdl, "params", n_par)
    opt = torch.optim.AdamW(mdl.parameters(), a.lr, weight_decay=1e-4)
    log, t0, best = [], time.time(), 0
    for it in range(1, a.its + 1):
        lr = a.lr * 0.5 * (1 + np.cos(np.pi * it / a.its))
        for g in opt.param_groups:
            g["lr"] = lr
        x = crops(ims, a.bs, a.p, gen)
        sig = torch.rand(a.bs, 1, 1, 1, device=d) * a.smax / 255
        y = x + torch.randn_like(x) * sig
        loss = F.l1_loss(mdl(y), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
        opt.step()
        if it % 500 == 0:
            v = val(mdl, vpairs, d)
            log.append({"it": it, "min": (time.time() - t0) / 60, "loss": loss.item(), "val25": v})
            print(log[-1], flush=True)
            if v > best:
                best = v
                torch.save(mdl.state_dict(), here / "ckpt" / f"{a.mdl}.pt")
    dump(
        {"mdl": a.mdl, "params": n_par, "args": vars(a), "best_val25": best, "log": log},
        here / "results" / f"train_{a.mdl}.json",
    )


if __name__ == "__main__":
    (here / "ckpt").mkdir(exist_ok=True)
    main()
