#!/usr/bin/env bash
# Break the alpha tie on unused development items, then read the matching test score.
set -euo pipefail
export PATH="/root/miniconda3/bin:${PATH}"
export PYTHONUNBUFFERED=1
ROOT="/root/autodl-tmp/option-mismatch-finetuning"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
MODEL="/root/autodl-tmp/models/Qwen2.5-0.5B-Instruct"
OUT="$ROOT/results/decord"

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
rows = [json.loads(line) for line in Path("data/processed/aqua_dev.jsonl").read_text().splitlines() if line.strip()]
tail = rows[92:]
path = Path("data/processed/aqua_dev_tail.jsonl")
path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in tail))
print(f"[data] tail {len(tail)}")
PY

python scripts/experiments/eval_decord.py \
  --model-dir "$MODEL" \
  --test-jsonl "$ROOT/data/processed/aqua_dev_tail.jsonl" \
  --max-new-tokens 256 \
  --decord-samples 1 \
  --recommit \
  --output-json "$OUT/dev_tail.json"

python scripts/experiments/select_tail.py \
  --tail-json "$OUT/dev_tail.json" \
  --output-json "$OUT/tail_choice.json"

rule=$(/root/miniconda3/bin/python -c 'import json; print(json.load(open("/root/autodl-tmp/option-mismatch-finetuning/results/decord/tail_choice.json"))["rule"])')
echo "[frozen] $rule"
if [[ "$rule" == "recommit" ]]; then
  python scripts/experiments/eval_decord.py \
    --model-dir "$MODEL" \
    --test-jsonl "$ROOT/data/processed/aqua_test.jsonl" \
    --max-new-tokens 256 \
    --decord-samples 1 \
    --recommit \
    --output-json "$OUT/test_base_recommit.json"
fi

/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
out = Path("/root/autodl-tmp/option-mismatch-finetuning/results/decord")
choice = json.loads((out / "tail_choice.json").read_text())
if choice["rule"] == "phase2":
    test = json.loads((out / "test_base.json").read_text())
    acc = float(test["decord"][choice["alpha"]]["accuracy"])
else:
    test = json.loads((out / "test_base_recommit.json").read_text())
    acc = float(test["recommit"]["accuracy"])
bar = 0.2637795275590551 + 0.01
payload = {
    "rule": choice["rule"],
    "alpha": choice["alpha"],
    "tail_accuracy": choice["tail_accuracy"],
    "test_accuracy": acc,
    "sft_accuracy": 0.2637795275590551,
    "dpo_accuracy": 0.1968503937007874,
    "base_accuracy": 0.16929133858267717,
    "decord_ft_accuracy": 0.31496062992125984,
    "bar": bar,
    "decord_beats": acc >= bar,
    "decord_ft_beats": True,
    "goal_met": acc >= bar,
}
(out / "gate_tail.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY
