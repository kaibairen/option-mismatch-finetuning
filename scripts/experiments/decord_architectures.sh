#!/usr/bin/env bash
set -euo pipefail
export PATH="/root/miniconda3/bin:${PATH}"
export PYTHONUNBUFFERED=1
ROOT="/root/autodl-tmp/option-mismatch-finetuning"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
python scripts/experiments/eval_architectures.py \
  --model-dir /root/autodl-tmp/models/Qwen2.5-0.5B-Instruct \
  --test-jsonl "$ROOT/data/processed/aqua_test.jsonl" \
  --max-new-tokens 256 \
  --output-json "$ROOT/results/decord/architectures_test.json"
