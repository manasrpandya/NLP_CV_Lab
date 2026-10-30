# 06 · text-based emotion classification

multi-label emotion classification on GoEmotions (58k reddit comments, 27 emotions + neutral). a roberta-large
teacher is distilled into distilroberta-base, which is exported to ONNX (int8) and runs in the browser.

- live demo: https://nlp-cv-lab.vercel.app/p/emotion-classification
- report: [report/report.pdf](report/report.pdf)
- models: [goemotions-distil-onnx](https://huggingface.co/manasrpandya/goemotions-distil-onnx) (deployed),
  [goemotions-teacher](https://huggingface.co/manasrpandya/goemotions-teacher),
  [goemotions-student](https://huggingface.co/manasrpandya/goemotions-student) (pytorch),
  logits for every run in [goemotions-runs](https://huggingface.co/datasets/manasrpandya/goemotions-runs)

## results

GoEmotions test set, 28 labels. every model gets per-label thresholds and a temperature tuned on the
validation set. ood = zero-shot accuracy on the dair-ai/emotion twitter test set after mapping both label
sets to ekman groups.

<!-- res -->
| model | macro-f1 | micro-f1 | ekman macro-f1 | ood acc (twitter) | ece raw / cal | size (mb) |
|---|---|---|---|---|---|---|
| tf-idf + logreg | 0.470 | 0.549 | 0.592 | 0.453 | 0.0743 / 0.0506 |  |
| distilroberta + kd | 0.534 | 0.611 | 0.646 | 0.632 | 0.0295 / 0.0098 |  |
| distilroberta + kd, int8 onnx | 0.543 | 0.612 | 0.646 | 0.631 | 0.0294 / 0.0097 |  |
| roberta-large (bce) | 0.542 | 0.616 | 0.653 | 0.621 | 0.0040 / 0.0042 |  |
| roberta-large (asl) | 0.547 | 0.610 | 0.647 | 0.627 | 0.2144 / 0.0618 |  |
| roberta-large ens. | 0.547 | 0.616 | 0.659 | 0.624 | 0.0281 / 0.0260 |  |
<!-- /res -->

## layout

```
src/emo/labels.py     label names, ekman grouping, dair-ai mapping
src/emo/data.py       dataset loading, tokenized dataset + padding collate
src/emo/losses.py     bce, asymmetric loss, binary kd loss
src/emo/train.py      transformer fine-tuning (teacher and student), llrd, fp16
src/emo/rnn.py        bigru + glove baseline
src/emo/tfidf.py      tf-idf + one-vs-rest logistic regression
src/emo/metrics.py    f1, threshold tuning, temperature scaling, ece
src/emo/evaluate.py   all runs' logits -> results/metrics.json
src/emo/export.py     optimum onnx export, int8 / fp16, parity, hub push
src/emo/bench.py      cpu latency + size
src/emo/figs.py       report figures
src/emo/tables.py     readme / report / site tables from the json
scripts/colab.sh      the training queue that ran on colab
colab/train.ipynb     same thing as a notebook
```

## how to run

```sh
uv sync
uv run pytest -q

# training (gpu, i used a colab t4: open colab/train.ipynb, or run scripts/colab.sh)
bash scripts/colab.sh

# everything after training runs on cpu
uv run python -m emo.tfidf
hf download manasrpandya/goemotions-runs --repo-type dataset --local-dir runs
uv run python -m emo.evaluate
uv run python -m emo.export && uv run python -m emo.evaluate
uv run python -m emo.bench
uv run python -m emo.figs
uv run python -m emo.tables
uv run python -m emo.export --push
cd report && tectonic report.tex
```
