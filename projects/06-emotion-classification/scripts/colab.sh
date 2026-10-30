#!/usr/bin/env bash
# full training queue on a colab t4. every run pushes its logits (and weights) to the hub when it finishes,
# and runs already on the hub are pulled instead of retrained, so after a disconnect just run it again.
set -euo pipefail
cd "$(dirname "$0")/.."
runs=manasrpandya/goemotions-runs
t="python -m emo.train --out runs --runs-repo $runs"

have() {
  hf download $runs --repo-type dataset --local-dir runs --include "$1/*" -q >/dev/null 2>&1 || true
  [ -f "runs/$1/logits.npz" ] && echo "skip $1 (already on the hub)"
}

have teacher-bce || $t --name teacher-bce --mdl roberta-large --loss bce --lr 2e-5 --llrd 0.95 --ep 4 \
  --train-logits --push manasrpandya/goemotions-teacher
have teacher-asl || $t --name teacher-asl --mdl roberta-large --loss asl --lr 2e-5 --llrd 0.95 --ep 4 \
  --train-logits --push manasrpandya/goemotions-teacher-asl
have student-kd || $t --name student-kd --mdl distilroberta-base --lr 6e-5 --llrd 0.9 --ep 6 \
  --kd runs/teacher-bce/logits.npz runs/teacher-asl/logits.npz --push manasrpandya/goemotions-student
have student-hard || $t --name student-hard --mdl distilroberta-base --lr 6e-5 --llrd 0.9 --ep 6
have bigru || python -m emo.rnn --out runs --runs-repo $runs
echo all done
