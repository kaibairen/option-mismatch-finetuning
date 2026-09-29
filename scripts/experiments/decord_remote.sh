#!/usr/bin/env bash
# DECORD campaign on the 2080 Ti. Smoke first, then the full AQUA comparison.
set -euo pipefail
export PATH="/root/miniconda3/bin:${PATH}"
export PYTHONUNBUFFERED=1
ROOT="/root/autodl-tmp/option-mismatch-finetuning"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
MODEL="/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"
CFG="$ROOT/configs/qwen25_0.5b_2080.yaml"
OUT="$ROOT/results/decord"
mkdir -p "$OUT" "$ROOT/results/adapters" "$ROOT/data/processed" "$ROOT/data/raw"

STAGE="${1:-smoke}"

python scripts/data/prepare_data.py --split test --output-name aqua_test
python scripts/data/prepare_data.py --split dev --output-name aqua_dev
python scripts/data/prepare_data.py --split train --sample 400 --seed 42 --output-name aqua_train_400
python scripts/data/prepare_data.py --split dev --sample 60 --seed 42 --output-name aqua_dev_60
python scripts/data/prepare_data.py --split train --sample 16 --seed 7 --output-name aqua_train_16
python scripts/data/prepare_data.py --split dev --sample 16 --seed 7 --output-name aqua_dev_16

run_eval() {
  local name="$1"
  local jsonl="$2"
  local adapter="${3:-}"
  local tokens="${4:-160}"
  if [[ -f "$OUT/${name}.json" ]]; then
    echo "[skip eval] $name"
    return 0
  fi
  python scripts/experiments/eval_decord.py \
    --model-dir "$MODEL" \
    --adapter "$adapter" \
    --test-jsonl "$jsonl" \
    --max-new-tokens "$tokens" \
    --output-json "$OUT/${name}.json"
}

run_train() {
  local method="$1"
  local dest="$2"
  shift 2
  if [[ -f "$dest/adapter_config.json" ]]; then
    echo "[skip train] $method"
    return 0
  fi
  python -m option_mismatch.train_decord \
    --config "$CFG" \
    --method "$method" \
    --local-model-dir "$MODEL" \
    --output-dir "$dest" \
    --max-length 512 \
    "$@"
}

if [[ "$STAGE" == "smoke" ]]; then
  run_eval smoke_base "$ROOT/data/processed/aqua_dev_16.jsonl" "" 160
  run_train sft "$ROOT/results/adapters/smoke_sft" \
    --train-jsonl "$ROOT/data/processed/aqua_train_16.jsonl" --epochs 1 --lr 1e-4
  run_train decord_sft "$ROOT/results/adapters/smoke_decord_sft" \
    --train-jsonl "$ROOT/data/processed/aqua_train_16.jsonl" --epochs 1 --lr 1e-4
  run_eval smoke_sft "$ROOT/data/processed/aqua_dev_16.jsonl" "$ROOT/results/adapters/smoke_sft" 160
  run_eval smoke_decord_sft "$ROOT/data/processed/aqua_dev_16.jsonl" "$ROOT/results/adapters/smoke_decord_sft" 160
  python - <<'PY'
import json
from pathlib import Path
out = Path("/root/autodl-tmp/option-mismatch-finetuning/results/decord")
for name in ("smoke_base", "smoke_sft", "smoke_decord_sft"):
    report = json.loads((out / f"{name}.json").read_text())
    print(name, "greedy", report["greedy"]["accuracy"], "decord", {k: v["accuracy"] for k, v in report["decord"].items()})
PY
  exit 0
fi

run_eval dev_base "$ROOT/data/processed/aqua_dev_60.jsonl" "" 160
run_train sft "$ROOT/results/adapters/sft_uniform" \
  --train-jsonl "$ROOT/data/processed/aqua_train_400.jsonl" --epochs 1 --lr 1e-4
if [[ ! -f "$ROOT/data/processed/aqua_pref_400.jsonl" ]]; then
  python -m option_mismatch.train_decord \
    --config "$CFG" \
    --method prefs \
    --local-model-dir "$MODEL" \
    --init-adapter "$ROOT/results/adapters/sft_uniform" \
    --train-jsonl "$ROOT/data/processed/aqua_train_400.jsonl" \
    --output-dir "$ROOT/data/processed/aqua_pref_400.jsonl" \
    --max-new-tokens 160
fi
run_train dpo "$ROOT/results/adapters/dpo_uniform" \
  --init-adapter "$ROOT/results/adapters/sft_uniform" \
  --pref-jsonl "$ROOT/data/processed/aqua_pref_400.jsonl" \
  --epochs 1 --lr 5e-5
run_train decord_sft "$ROOT/results/adapters/decord_sft" \
  --train-jsonl "$ROOT/data/processed/aqua_train_400.jsonl" --epochs 1 --lr 1e-4
run_train decord_dpo "$ROOT/results/adapters/decord_ft" \
  --init-adapter "$ROOT/results/adapters/decord_sft" \
  --pref-jsonl "$ROOT/data/processed/aqua_pref_400.jsonl" \
  --epochs 1 --lr 5e-5

run_eval dev_decord_sft "$ROOT/data/processed/aqua_dev_60.jsonl" "$ROOT/results/adapters/decord_sft" 160
run_eval dev_decord_ft "$ROOT/data/processed/aqua_dev_60.jsonl" "$ROOT/results/adapters/decord_ft" 160
run_eval test_base "$ROOT/data/processed/aqua_test.jsonl" "" 160
run_eval test_sft "$ROOT/data/processed/aqua_test.jsonl" "$ROOT/results/adapters/sft_uniform" 160
run_eval test_dpo "$ROOT/data/processed/aqua_test.jsonl" "$ROOT/results/adapters/dpo_uniform" 160
run_eval test_decord_sft "$ROOT/data/processed/aqua_test.jsonl" "$ROOT/results/adapters/decord_sft" 160
run_eval test_decord_ft "$ROOT/data/processed/aqua_test.jsonl" "$ROOT/results/adapters/decord_ft" 160

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
