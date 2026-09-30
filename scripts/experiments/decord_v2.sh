#!/usr/bin/env bash
# Rescore DECORD with end-of-trace numbers when no answer line exists.
# The rule is frozen on dev before the test split is touched.
set -euo pipefail
export PATH="/root/miniconda3/bin:${PATH}"
export PYTHONUNBUFFERED=1
ROOT="/root/autodl-tmp/option-mismatch-finetuning"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
MODEL="/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"
OUT="$ROOT/results/decord"

python scripts/experiments/eval_decord.py \
  --model-dir "$MODEL" \
  --test-jsonl "$ROOT/data/processed/aqua_dev_60.jsonl" \
  --max-new-tokens 256 \
  --decord-samples 1 \
  --output-json "$OUT/dev_base_v2.json"

python scripts/experiments/select_commitment.py \
  --dev-json "$OUT/dev_base_v2.json" \
  --output-json "$OUT/commitment_choice.json"

rule=$(/root/miniconda3/bin/python -c 'import json; print(json.load(open("'"$OUT"'/commitment_choice.json"))["rule"])')
echo "[choice] $rule"
if [[ "$rule" != "fullnum" ]]; then
  echo "[stop] full-trace numbers did not beat the phase-2 rule on dev"
  exit 0
fi

python scripts/experiments/eval_decord.py \
  --model-dir "$MODEL" \
  --test-jsonl "$ROOT/data/processed/aqua_test.jsonl" \
  --max-new-tokens 256 \
  --decord-samples 1 \
  --output-json "$OUT/test_base_v2.json"

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
out = Path("/root/autodl-tmp/option-mismatch-finetuning/results/decord")
choice = json.loads((out / "commitment_choice.json").read_text())
test = json.loads((out / "test_base_v2.json").read_text())
block = test["decord_fullnum"][choice["alpha"]]
acc = float(block["accuracy"])
bar = 0.2637795275590551 + 0.01
payload = {
    "rule": choice["rule"],
    "alpha": choice["alpha"],
    "test_accuracy": acc,
    "bar": bar,
    "decord_beats": acc >= bar,
    "decord_ft_accuracy": 0.31496062992125984,
    "decord_ft_beats": True,
    "goal_met": acc >= bar,
}
(out / "gate_v2.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY
