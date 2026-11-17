#!/usr/bin/env bash
# training + generation queue for a colab t4. every run pushes its outputs to the hub as it finishes and
# runs already on the hub are skipped, so after a disconnect just run it again.
set -uo pipefail
cd "$(dirname "$0")/.."
runs=manasrpandya/cnndm-summ-runs
g="python -m summ.gen --runs-repo $runs"
t="python -m summ.train --runs-repo $runs"

have() {
  hf download $runs --repo-type dataset --local-dir runs --include "$1/*" -q >/dev/null 2>&1 || true
  [ -f "runs/$1/$2" ] && echo "skip $1 (already on the hub)"
}
gen() { have "$1" gens.jsonl || $g --name "$@"; }
trn() { have "$1" log.json || $t --name "$@"; }

# our model (bart-base + length prefix)
trn ours --mdl facebook/bart-base --n 40000 --push manasrpandya/cnndm-summ
gen ours-medium-b4 --mdl manasrpandya/cnndm-summ --prefix medium
gen ours-medium-greedy --mdl manasrpandya/cnndm-summ --prefix medium --beams 1 --bs 32
gen ours-short-b4 --mdl manasrpandya/cnndm-summ --prefix short
gen ours-long-b4 --mdl manasrpandya/cnndm-summ --prefix long

# pretrained references on the fixed 2000-article test subset (pegasus in fp32 is slow, so only the first 500)
gen bart-large-cnn --mdl facebook/bart-large-cnn --hub-defaults --max-src 1024 --bs 16
gen pegasus-cnndm --mdl google/pegasus-cnn_dailymail --hub-defaults --max-src 1024 --fp32 --bs 8 --head 500
gen distilbart-cnn --mdl sshleifer/distilbart-cnn-12-6 --hub-defaults --max-src 1024 --bs 16


# sequence-level distillation: teacher summaries for 8k train articles, then the same budget with gold targets
gen teacher-train8k --mdl facebook/bart-large-cnn --hub-defaults --max-src 1024 --beams 2 --bs 32 --split train --perm --n 8000
trn ours-kd8k --mdl facebook/bart-base --n 8000 --targets runs/teacher-train8k/gens.jsonl --push manasrpandya/cnndm-summ-kd8k
trn ours-gold8k --mdl facebook/bart-base --n 8000 --push manasrpandya/cnndm-summ-gold8k
gen kd8k-medium-b4 --mdl manasrpandya/cnndm-summ-kd8k --prefix medium
gen gold8k-medium-b4 --mdl manasrpandya/cnndm-summ-gold8k --prefix medium

# xsum as the second set
trn xsum-ours --mdl facebook/bart-base --data xsum --n 20000 --plain --max-tgt 64 --push manasrpandya/xsum-summ
gen xsum-ours --mdl manasrpandya/xsum-summ --data xsum --n 1000 --plain --max-len 64 --lp 1.0 --bs 32
gen xsum-bart-large --mdl facebook/bart-large-xsum --data xsum --n 1000 --hub-defaults --max-src 1024 --bs 16
echo all done
