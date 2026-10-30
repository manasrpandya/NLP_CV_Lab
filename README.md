# nlp & cv lab

lab tasks for the nlp and computer vision course. everything is pytorch + hugging face, and each trained
model is exported to onnx so it runs directly in the browser on the demo site.

**live demos:** https://nlp-cv-lab.vercel.app

| # | task | status | report | demo |
|---|------|--------|--------|------|
| 01 | image caption generation | todo | | |
| 02 | seq2seq translation | todo | | |
| 03 | text summarization | todo | | |
| 04 | question answering | todo | | |
| 05 | grammar error correction | todo | | |
| 06 | text emotion classification | live | [pdf](projects/06-emotion-classification/report/report.pdf) | [demo](https://nlp-cv-lab.vercel.app/p/emotion-classification) |
| 07 | bilateral image denoising | live | [pdf](projects/07-bilateral-denoising/report/report.pdf) | [demo](https://nlp-cv-lab.vercel.app/p/bilateral-denoising) |
| 08 | eye detection & facial landmarks | todo | | |
| 09 | human activity recognition | todo | | |
| 10 | vehicle detection & counting | todo | | |
| 11 | visual question answering | todo | | |
| 12 | table detection & extraction | todo | | |
| 13 | signboard text recognition & translation | todo | | |
| 14 | product recognition & description | todo | | |
| 15 | multimodal sentiment analysis | todo | | |

## layout

```
packages/labcore/   shared helpers (seeding, timing, json results, onnx parity check)
projects/NN-name/   one folder per task: code, configs, tests, results, report
web/                next.js site, one page per task under /p/<slug>
```

each project has its own `pyproject.toml` and venv (managed with uv), so dependencies don't clash between
tasks. trained weights are pushed to the hugging face hub instead of git.

## running a project

```sh
cd projects/07-bilateral-denoising
uv sync
uv run python -m <pkg>.train --help
```

## site

```sh
cd web
pnpm i
pnpm dev
```
