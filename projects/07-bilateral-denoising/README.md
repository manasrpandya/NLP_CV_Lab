# 07 · image denoising with bilateral filters

lab task 07. we start from the plain bilateral filter (tomasi & manduchi 1998), tune it properly, make it
pick its own parameters from a blind noise estimate, and then train a tiny network that predicts the
bilateral parameters per pixel ("neural-guided bilateral", nbf). everything is compared against NLM, BM3D
and a pretrained DnCNN so it's clear where bilateral filtering stands.

- live demo: https://nlp-cv-lab.vercel.app/p/bilateral-denoising (webgl bilateral that re-filters as you
  move the sliders, auto mode, and the nbf model running in the browser with onnx runtime)
- model: https://huggingface.co/manasrpandya/bilateral-denoise-nbf (onnx, 462 kB)
- report: [report/report.pdf](report/report.pdf)
- code: https://github.com/manasrpandya/NLP_CV_Lab/tree/main/projects/07-bilateral-denoising

## results

PSNR (dB) on grayscale images, gaussian noise with a fixed seed, plus 320 real-noise crops from the SIDD
validation set. (a) to (d) are bilateral variants, (e) are reference denoisers. all numbers come from
`results/eval/*.json`.

<!-- tab -->
| | method | Set12 15 | 25 | 50 | BSD68 15 | 25 | 50 | Kodak24 15 | 25 | 50 | SIDD |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| - | noisy input | 24.69 | 20.32 | 14.76 | 24.79 | 20.48 | 14.91 | 24.74 | 20.39 | 14.78 | 27.02 |
| a | bilateral, tuned per sigma | 30.35 | 27.44 | 24.03 | 29.71 | 26.96 | 23.98 | 30.38 | 27.54 | 24.57 | 30.47 |
| b | bilateral, blind auto params | 30.29 | 27.39 | 24.03 | 29.60 | 26.90 | 23.97 | 30.31 | 27.48 | 24.56 | 30.47 |
| c | iterated bilateral | 30.45 | 27.82 | 24.57 | 29.81 | 27.20 | 24.36 | 30.62 | 28.00 | 24.97 | 30.83 |
| c | joint bilateral (smoothed guide) | 30.88 | 28.17 | 24.35 | 30.00 | 27.45 | 24.05 | 30.81 | 28.15 | 24.54 | 30.75 |
| d | neural-guided bilateral (ours) | **31.93** | **29.60** | **26.44** | **30.82** | **28.38** | **25.48** | **32.11** | **29.79** | **26.84** | **27.27** |
| e | non-local means (cv2) | 30.36 | 27.82 | 24.28 | 29.37 | 26.68 | 23.69 | 30.37 | 27.65 | 24.61 | 32.59 |
| e | BM3D | 32.40 | 30.00 | 26.73 | 31.15 | 28.63 | 25.71 | 32.37 | 29.96 | 27.02 | 28.21 |
| e | DnCNN-B (pretrained) | 32.68 | 30.36 | 27.18 | 31.62 | 29.16 | 26.23 | 32.78 | 30.42 | 27.51 | 27.48 |
<!-- /tab -->

the short version: tuning the bilateral filter per noise level gets about +6.5 dB over the noisy input at
sigma 25, the blind auto version loses about 0.1 dB of that, and joint/iterated filtering add another
0.2-0.6 dB. letting a 93k parameter unet choose sig_s, sig_r and the range guide per pixel adds about
1.4 dB more on BSD68 and gets within 0.2-0.5 dB of BM3D, while the output is still a normalised average of
noisy neighbours. it does not beat DnCNN (666k params), and on real SIDD noise the gaussian-trained
networks (ours and DnCNN) barely do anything, while the classical filters driven by a noise estimate do
much better. one honest ablation result: the same unet trained to predict the noise directly is just as
good as nbf (within 0.03 dB), so the bilateral structure doesn't cost PSNR but doesn't add any either. what
it buys is interpretable per-pixel parameter maps, an output that can't invent texture, and faster early
training. see the report for the details.

## layout

```
src/bden/
  data.py      download + cache BSD400 / Set12 / BSD68 / Kodak24 / SIDD val, fixed-seed noise
  bf.py        differentiable bilateral layer (per-pixel sig_s / sig_r maps, optional guide)
  est.py       blind noise estimators (immerkaer, wavelet mad)
  search.py    grid search of cv2 bilateral params on held-out val, noise estimator calibration
  tune.py      param search for iterated / joint bilateral and NLM
  classic.py   classical methods (a, b, c, e)
  model.py     nbf (unet + two bilateral passes) and the ablations
  train.py     training (blind, sigma in [0, 55])
  evaluate.py  psnr / ssim / runtime per method, dataset and sigma -> results/eval/
  bench.py     runtime on one 512x512 image -> results/bench.json
  export.py    onnx export + parity check against pytorch
  push.py      upload onnx + model card to the hf hub
  figs.py      all report figures -> report/figs/
  tables.py    report tables + the table in this readme
  site.py      compact results json for the demo page
tests/         bilateral layer vs cv2, noise estimator, model shapes, onnx parity
```

## how to run

needs python 3.12 and [uv](https://docs.astral.sh/uv/). on linux uv pulls cuda torch, it also runs on cpu
(just slower).

```sh
uv sync --extra ref                     # ref = bm3d, only needed for the reference rows
uv run python -m bden.data              # downloads ~350 MB into data/
uv run python -m bden.search            # bilateral grid search + estimator calibration (~2 min)
uv run python -m bden.tune
uv run python -m bden.train --mdl nbf   # ~10-15 min on a laptop gpu (rtx 2050)
uv run python -m bden.evaluate --methods noisy bf_oracle bf_auto bf_iter bf_joint nlm bm3d nbf dncnn
uv run python -m bden.export && uv run python -m bden.bench
uv run python -m bden.figs && uv run python -m bden.tables
uv run pytest -q
cd report && tectonic report.tex
```

