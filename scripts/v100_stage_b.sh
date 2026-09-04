#!/usr/bin/env bash
# Stage B: real preferences + SFT / DPO / Rep-DPO on 0.5B, eval on AQUA test-100.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WORK="${WORK_ROOT:-/root/autodl-tmp}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_ROOT="${MODEL_ROOT:-$WORK/models}"
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate base
fi
PY="${PYTHON:-python}"
CFG="$ROOT/configs/qwen25_0.5b_2080.yaml"
MODEL="$MODEL_ROOT/Qwen2.5-0.5B-Instruct"
PREF="$ROOT/data/processed/aqua_pref_100.jsonl"

"$PY" - <<PY
from option_mismatch.preferences import build_preferences
build_preferences("$CFG", {
    "local_model_dir": "$MODEL",
    "train_jsonl": "$ROOT/data/processed/aqua_train_100.jsonl",
    "max_train_samples": 100,
    "output_jsonl": "$PREF",
})
print("preferences", "$PREF")
PY

for method in sft dpo rep_dpo; do
  "$PY" -m option_mismatch.train_rep_dpo \
    --config "$CFG" \
    --method "$method" \
    --pref-jsonl "$PREF" \
    --output-dir "$ROOT/results/adapters/${method}_0.5b"
done

"$PY" -m option_mismatch.eval_mcq --config "$CFG" --output-json "$ROOT/results/eval_base.json"
"$PY" -m option_mismatch.eval_mcq --config "$CFG" --adapter "$ROOT/results/adapters/sft_0.5b" --output-json "$ROOT/results/eval_sft.json"
"$PY" -m option_mismatch.eval_mcq --config "$CFG" --adapter "$ROOT/results/adapters/dpo_0.5b" --output-json "$ROOT/results/eval_dpo.json"
"$PY" -m option_mismatch.eval_mcq --config "$CFG" --adapter "$ROOT/results/adapters/rep_dpo_0.5b" --output-json "$ROOT/results/eval_rep_dpo.json"
"$PY" "$ROOT/scripts/merge_stage_b.py"
echo "[stage-b] done"
