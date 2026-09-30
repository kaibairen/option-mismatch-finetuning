"""Score the frozen full-trace rule at alpha 2.0. No other alpha is eligible."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("/root/autodl-tmp/option-mismatch-finetuning/results/decord")
test = json.loads((OUT / "test_base_v2.json").read_text(encoding="utf-8"))
acc = float(test["decord_fullnum"]["2.0"]["accuracy"])
bar = 0.2637795275590551 + 0.01
payload = {
    "rule": "fullnum",
    "alpha": "2.0",
    "tail_accuracy": 0.25925925925925924,
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
(OUT / "gate_fullnum.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
