#!/usr/bin/env bash
# full training queue on a colab t4. every run pushes its logits (and weights) to the hub when it finishes,
# so a disconnect only loses the run in progress.
set -euo pipefail
cd "$(dirname "$0")/.."
runs=manasrpandya/goemotions-runs
t="python -m emo.train --out runs --runs-repo $runs"

$t --name teacher-bce --mdl roberta-large --loss bce --lr 2e-5 --llrd 0.95 --ep 4 --train-logits \
  --push manasrpandya/goemotions-teacher
$t --name teacher-asl --mdl roberta-large --loss asl --lr 2e-5 --llrd 0.95 --ep 4 --train-logits \
  --push manasrpandya/goemotions-teacher-asl
$t --name student-kd --mdl distilroberta-base --lr 6e-5 --llrd 0.9 --ep 6 \
  --kd runs/teacher-bce/logits.npz runs/teacher-asl/logits.npz --push manasrpandya/goemotions-student
$t --name student-hard --mdl distilroberta-base --lr 6e-5 --llrd 0.9 --ep 6
python -m emo.rnn --out runs --runs-repo $runs
echo all done
