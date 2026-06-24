#!/usr/bin/env bash
# One-command summary of the v1.5 real-model margin-widening sweep.
# Aggregates the GATE (||r|| cv) + HEADROOM (bstar) over a directory of fieldrun --source-dump files.
#
# Upstream (once, per model; needs a built fieldrun + HF cache — NOT a code dependency, a dev convenience):
#   fieldrun convert --model <hf-id> --arch <neox|rope> --dtype f16 -o bundles/<m>
#   fieldrun --bundle bundles/<m> --recursion-explain --source-dump <DUMPDIR>/<m>_source.jsonl --n 220
#
# Then:  experiments/margin_widening_sweep.sh <DUMPDIR>
set -euo pipefail
DIR="${1:?usage: margin_widening_sweep.sh <dir-of-*_source.jsonl>}"
PY="${PY:-.venv/bin/python}"
printf "%-26s %5s %4s %8s %10s %10s %s\n" "dump" "dim" "N" "rcv" "bstar50" "bstar90" "gate"
for f in "$DIR"/*_source.jsonl "$DIR"/*.jsonl; do
  [ -e "$f" ] || continue
  lbl=$(basename "$f" | sed 's/_source\.jsonl$//;s/\.jsonl$//')
  PYTHONPATH=. "$PY" experiments/margin_widening_realdata.py "$f" "$lbl" --gate-only 2>/dev/null \
    | grep "^SWEEP" | awk -F'\t' '{printf "%-26s %5s %4s %8s %10s %10s   %s\n",$2,$3,$4,$5,$6,$7,$9}'
done
echo "gate CLOSED (rcv << 0.35) everywhere => final-norm pins ||r|| => widen_t == raw (normalized term refuted)."
