#!/usr/bin/env bash
# Preflight, then merge. Refuses incompatible checkpoints by default.
set -euo pipefail

CONFIG="${1:-configs/qwen36-nemotron.yaml}"
OUT="${2:-./merged}"

echo "[*] Preflight: are these checkpoints the same base architecture?"
if ! python scripts/preflight.py --config "$CONFIG"; then
  echo
  echo "[!] Refusing to merge: the configs don't line up (see verdict above)."
  echo "    Element-wise merging incompatible models is undefined. If you really"
  echo "    mean it, add --allow-crimes to the mergekit-yaml call below yourself."
  exit 1
fi

echo
echo "[*] Merging per $CONFIG -> $OUT"
mergekit-yaml "$CONFIG" "$OUT" \
  --cuda \
  --lazy-unpickle \
  --out-shard-size 5B \
  --copy-tokenizer

echo "[*] Done. Merged weights at $OUT"
echo "    Next: bash scripts/evaluate.sh $OUT results"
