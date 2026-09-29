#!/usr/bin/env bash
# Score the frozen adapters on the full AQUA-RAT test split. One sample per item.
set -euo pipefail
export PATH="/root/miniconda3/bin:${PATH}"
export PYTHONUNBUFFERED=1
ROOT="/root/autodl-tmp/option-mismatch-finetuning"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
MODEL="/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"
TEST="$ROOT/data/processed/aqua_test.jsonl"
OUT="$ROOT/results/decord"

run_eval() {
  local name="$1"
  local adapter="${2:-}"
  if [[ -f "$OUT/${name}.json" ]]; then
    echo "[skip] $name"
    return 0
  fi
  python scripts/experiments/eval_decord.py \
    --model-dir "$MODEL" \
    --adapter "$adapter" \
    --test-jsonl "$TEST" \
    --max-new-tokens 256 \
    --decord-samples 1 \
    --output-json "$OUT/${name}.json"
}

run_eval test_base ""
run_eval test_sft "$ROOT/results/adapters/sft_uniform"
run_eval test_dpo "$ROOT/results/adapters/dpo_uniform"
run_eval test_decord_sft "$ROOT/results/adapters/decord_sft"
run_eval test_decord_ft "$ROOT/results/adapters/decord_ft"

python scripts/experiments/decord_gate.py \
  --dev-base "$OUT/dev_base.json" \
  --dev-decord-sft "$OUT/dev_decord_sft.json" \
  --dev-decord-ft "$OUT/dev_decord_ft.json" \
  --test-base "$OUT/test_base.json" \
  --test-sft "$OUT/test_sft.json" \
  --test-dpo "$OUT/test_dpo.json" \
  --test-decord-sft "$OUT/test_decord_sft.json" \
  --test-decord-ft "$OUT/test_decord_ft.json" \
  --output-json "$OUT/gate.json"
