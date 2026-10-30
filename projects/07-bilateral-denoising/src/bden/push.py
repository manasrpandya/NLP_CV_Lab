from pathlib import Path

from huggingface_hub import HfApi
from labcore import load

from bden.evaluate import load_eval

here = Path(__file__).resolve().parents[2]
repo = "manasrpandya/bilateral-denoise-nbf"

card = """---
license: mit
library_name: onnx
tags: [image-denoising, bilateral-filter, onnx]
---

# neural-guided bilateral filter (nbf)

tiny unet ({params} params) that predicts per-pixel sigma_s, sigma_r and a range guide for two passes of a
differentiable bilateral filter (7x7 disc window). the output is always a normalised weighted average of
noisy neighbours, so it behaves like a bilateral filter that picks its own parameters. trained blind on BSD400
patches with gaussian noise sigma in [0, 55] (grayscale, one channel, for rgb run each channel as
a batch item).

part of a university lab task, code and report:
https://github.com/manasrpandya/NLP_CV_Lab/tree/main/projects/07-bilateral-denoising
live demo (runs this file in the browser): https://nlp-cv-lab.vercel.app/p/bilateral-denoising

## io

- input `x`: float32 (n, 1, h, w) in [0, 1], any h, w
- output `y`: float32 (n, 1, h, w) denoised
- output `maps`: float32 (n, 5, h, w) = sigma_s pass 1, sigma_s pass 2, sigma_r pass 1, sigma_r pass 2, guide

## results (PSNR dB, gaussian noise, gray)

| set | s=15 | s=25 | s=50 |
|---|---|---|---|
{rows}

opset 17, {kb:.0f} kB, max abs diff vs pytorch {err:.1e}.
"""


def main():
    ev = load_eval()["nbf"]
    ox = load(here / "results" / "onnx.json")
    rows = "\n".join(
        f"| {ds} | " + " | ".join(f"{ev[f'{ds}/{s}']['psnr']:.2f}" for s in (15, 25, 50)) + " |"
        for ds in ("set12", "bsd68", "kodak24")
    )
    params = load(here / "results" / "train_nbf.json")["params"]
    md = card.format(params=f"{params / 1000:.0f}k", rows=rows, kb=ox["kb"], err=ox["max_abs_err"])
    api = HfApi()
    api.create_repo(repo, exist_ok=True)
    api.upload_file(path_or_fileobj=str(here / "ckpt" / "nbf.onnx"), path_in_repo="nbf.onnx", repo_id=repo)
    api.upload_file(path_or_fileobj=md.encode(), path_in_repo="README.md", repo_id=repo)
    print("pushed", repo)


if __name__ == "__main__":
    main()
