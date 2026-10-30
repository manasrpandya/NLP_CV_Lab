from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from labcore import dump, load  # noqa: E402

from bden import classic, data, zoo  # noqa: E402
from bden.est import immerkaer, wavelet  # noqa: E402
from bden.evaluate import cpu_fn  # noqa: E402
from bden.metrics import psnr  # noqa: E402

here = Path(__file__).resolve().parents[2]
out = here / "report" / "figs"
res = here / "results"
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
        "font.family": "serif",
    }
)


def nbf_run(y, name="nbf"):
    mdl = zoo.ours(name)
    with torch.no_grad():
        o, m = mdl(torch.from_numpy(y)[None, None], maps=True)
    return o[0, 0].numpy(), m[0].numpy()


def psnr_vs_sigma():
    ev = load(res / "eval.json")
    keys = [
        ("bf_oracle", "bilateral (a)"),
        ("bf_iter", "iterated (c)"),
        ("bf_joint", "joint (c)"),
        ("nbf", "neural-guided bilateral (d)"),
        ("cnn", "same unet, residual"),
        ("nlm", "NLM"),
        ("bm3d", "BM3D"),
        ("dncnn", "DnCNN-B"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.4), sharey=False)
    for ax, ds in zip(axs, data.test_sets):
        for k, lbl in keys:
            ys = [ev[k][f"{ds}/{s}"]["psnr"] for s in data.sigmas]
            st = (
                dict(lw=2, color="C3")
                if k == "nbf"
                else dict(lw=1, ls="--" if k in ("nlm", "bm3d", "dncnn", "cnn") else "-")
            )
            ax.plot(data.sigmas, ys, marker="o", ms=2.5, label=lbl, **st)
        ax.set_title(ds)
        ax.set_xlabel("noise sigma")
        ax.set_xticks(data.sigmas)
    axs[0].set_ylabel("PSNR (dB)")
    axs[-1].legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "psnr_sigma.pdf")


def crops():
    # set12 house at sigma 25 and a bsd68 image at 50: noisy / bilateral / joint / ours / bm3d / clean
    rows = [("set12", 25, 1, (60, 60, 140)), ("bsd68", 50, 2, (100, 150, 150))]
    cols = ["noisy", "bf_oracle", "bf_joint", "nbf", "bm3d", "clean"]
    names = {
        "noisy": "noisy",
        "bf_oracle": "bilateral",
        "bf_joint": "joint bil.",
        "nbf": "ours (nbf)",
        "bm3d": "BM3D",
        "clean": "clean",
    }
    fig, axs = plt.subplots(len(rows), len(cols), figsize=(7, 2.6))
    for i, (ds, sig, idx, (r0, c0, s)) in enumerate(rows):
        x, y = data.noisy(ds, sig)[idx]
        for j, m in enumerate(cols):
            im = x if m == "clean" else nbf_run(y)[0] if m == "nbf" else cpu_fn(m, y, sig)
            ax = axs[i, j]
            ax.imshow(np.clip(im[r0 : r0 + s, c0 : c0 + s], 0, 1), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            lbl = names[m] if m == "clean" else f"{names[m]}\n{psnr(x, im):.2f} dB"
            ax.set_title(lbl, fontsize=6.5)
        axs[i, 0].set_ylabel(f"{ds} #{idx}, s={sig}", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "crops.pdf")


def method_noise():
    x, y = data.noisy("set12", 25)[1]
    ms = [
        ("bf_oracle", "bilateral"),
        ("bf_joint", "joint bilateral"),
        ("nbf", "ours (nbf)"),
        ("bm3d", "BM3D"),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(7, 1.9))
    for ax, (m, lbl) in zip(axs, ms):
        d = nbf_run(y)[0] if m == "nbf" else cpu_fn(m, y, 25)
        ax.imshow(y - d, cmap="gray", vmin=-0.25, vmax=0.25)
        ax.set_title(f"{lbl}: noisy - output", fontsize=7)
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "method_noise.pdf")


def maps():
    x = data.noisy("set12", 25)[0][0]
    fig, axs = plt.subplots(2, 4, figsize=(7, 3.6))
    stats = []
    for i, sig in enumerate([15, 50]):
        y = (x + np.random.default_rng(3).normal(0, sig / 255, x.shape)).astype(np.float32)
        o, m = nbf_run(y)
        show = [
            (y, f"noisy, s={sig}", "gray", (0, 1)),
            (m[2] * 255, "sig_r pass 1", "magma", None),
            (m[0], "sig_s pass 1 (px)", "viridis", None),
            (m[4], "guide", "gray", (0, 1)),
        ]
        for ax, (im, t, cm, lim) in zip(axs[i], show):
            h = ax.imshow(im, cmap=cm, **({"vmin": lim[0], "vmax": lim[1]} if lim else {}))
            ax.set_title(t, fontsize=7)
            ax.axis("off")
            if lim is None:
                fig.colorbar(h, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=5)
    fig.tight_layout(pad=0.3)
    fig.savefig(out / "maps.pdf")

    # does the net infer the noise level blindly? mean predicted sig_r vs true sigma on bsd68
    for sig in range(0, 61, 5):
        vals = []
        for x, _ in data.noisy("bsd68", 25)[:20]:
            y = (x + np.random.default_rng(sig).normal(0, sig / 255, x.shape)).astype(np.float32)
            m = nbf_run(y)[1]
            vals.append([np.median(m[2]) * 255, np.median(m[3]) * 255, np.median(m[0]), np.median(m[1])])
        v = np.mean(vals, 0)
        stats.append({"sigma": sig, "sig_r1": v[0], "sig_r2": v[1], "sig_s1": v[2], "sig_s2": v[3]})
    dump(stats, res / "nbf_maps.json")
    fit = load(res / "auto_fit.json")["table"]
    fig, ax = plt.subplots(1, 2, figsize=(6, 2.2))
    sg = [s["sigma"] for s in stats]
    ax[0].plot(sg, [s["sig_r1"] for s in stats], "o-", ms=2.5, color="C3", label="nbf pass 1 (median)")
    ax[0].plot(sg, [s["sig_r2"] for s in stats], "o-", ms=2.5, color="C1", label="nbf pass 2 (median)")
    ax[0].plot(
        [t["sigma"] for t in fit],
        [t["sig_r"] for t in fit],
        "s--",
        ms=2.5,
        color="k",
        label="grid search best (a)",
    )
    ax[0].set_xlabel("true noise sigma")
    ax[0].set_ylabel("sig_r (0-255)")
    ax[0].legend(fontsize=6, frameon=False)
    ax[1].plot(sg, [s["sig_s1"] for s in stats], "o-", ms=2.5, color="C3", label="nbf pass 1")
    ax[1].plot(sg, [s["sig_s2"] for s in stats], "o-", ms=2.5, color="C1", label="nbf pass 2")
    ax[1].plot(
        [t["sigma"] for t in fit], [t["sig_s"] for t in fit], "s--", ms=2.5, color="k", label="grid (a)"
    )
    ax[1].set_xlabel("true noise sigma")
    ax[1].set_ylabel("sig_s (px)")
    ax[1].legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "params_sigma.pdf")


def estimator():
    cal = load(res / "auto_fit.json")["est"]
    rows = {}
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    for name, f, c in (("immerkaer", immerkaer, "C0"), ("wavelet", wavelet, "C2")):
        a, b = cal[name]["a"], cal[name]["b"]
        tru, est = [], []
        for ds in data.test_sets:
            for sig in data.sigmas:
                for _, y in data.noisy(ds, sig):
                    tru.append(sig)
                    est.append(a * f(y) * 255 + b)
        tru, est = np.array(tru), np.array(est)
        rows[name] = {
            str(s): {
                "mae": float(np.abs(est - tru)[tru == s].mean()),
                "bias": float((est - tru)[tru == s].mean()),
            }
            for s in data.sigmas
        }
        j = np.random.default_rng(0).normal(0, 0.6, len(tru))
        ax.scatter(tru + j + (0.8 if name == "wavelet" else -0.8), est, s=2, alpha=0.4, color=c, label=name)
    sidd = [classic.est_sigma(y, cal["immerkaer"]) for _, y in data.noisy("sidd", 0)]
    sidd_true = [np.std(y - x) * 255 for x, y in data.noisy("sidd", 0)]
    rows["sidd"] = {"est_median": float(np.median(sidd)), "true_std_median": float(np.median(sidd_true))}
    ax.plot([10, 55], [10, 55], "k:", lw=0.8)
    ax.set_xlabel("true sigma")
    ax.set_ylabel("estimated sigma")
    ax.legend(fontsize=6, frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(out / "estimator.pdf")
    dump(rows, res / "est.json")


def curves():
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for k, lbl in [
        ("nbf", "nbf"),
        ("nbf_noguide", "nbf, no guide"),
        ("nbf_1stage", "nbf, one pass"),
        ("cnn", "same unet, residual"),
    ]:
        lg = load(res / f"train_{k}.json")["log"]
        ax.plot([e["it"] for e in lg], [e["val25"] for e in lg], label=lbl, lw=1.8 if k == "nbf" else 1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("val PSNR, s=25")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "curves.pdf")


def main():
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    for f in (psnr_vs_sigma, crops, method_noise, maps, estimator, curves):
        f()
        print(f.__name__, "done", flush=True)


if __name__ == "__main__":
    main()
