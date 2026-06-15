#!/usr/bin/env bash
# Run the same benchmark suite on a model (parent or merged child).
# Usage: bash scripts/evaluate.sh <hf_id_or_path> [output_dir]
#   TASKS=...    override task list
#   BACKEND=vllm use vLLM instead of plain hf (much faster for big MoE)
set -euo pipefail

MODEL="${1:?usage: evaluate.sh <hf_id_or_path> [output_dir]}"
OUT="${2:-results}"
TASKS="${TASKS:-arc_challenge,hellaswag,winogrande,gsm8k,truthfulqa_mc2}"
BACKEND="${BACKEND:-hf}"

mkdir -p "$OUT"

if [ "$BACKEND" = "vllm" ]; then
  MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16,gpu_memory_utilization=0.9,trust_remote_code=True"
else
  MODEL_ARGS="pretrained=${MODEL},dtype=bfloat16,trust_remote_code=True"
fi

echo "[*] eval ${MODEL}  (backend=${BACKEND}, tasks=${TASKS})"
lm_eval \
  --model "$BACKEND" \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASKS" \
  --batch_size auto \
  --output_path "$OUT"

echo "[*] results written under ${OUT}/"
echo "    Tabulate with: python scripts/compare.py ${OUT}"
