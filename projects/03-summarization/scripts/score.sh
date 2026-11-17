#!/usr/bin/env bash
# score every generated run (rouge, bertscore, novelty, nli faithfulness) on the colab gpu and push the json
set -euo pipefail
cd "$(dirname "$0")/.."
runs=manasrpandya/cnndm-summ-runs
hf download $runs --repo-type dataset --local-dir runs -q >/dev/null
python -m summ.score lead3 textrank3 distilbart-cnn pegasus-cnndm bart-large-cnn \
  ours-medium-b4 ours-medium-greedy ours-short-b4 ours-long-b4 kd8k-medium-b4 gold8k-medium-b4 \
  xsum-lead1 xsum-lead3 xsum-textrank1 xsum-ours xsum-bart-large --res results/metrics.json
hf upload $runs results/metrics.json results/metrics.json --repo-type dataset
echo scored
