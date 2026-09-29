"""Apply the dev-selected DECORD alpha and check the +1 percentage-point gate on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy(report: dict, key: str, alpha: str | None = None) -> float:
    if key == "greedy":
        return float(report["greedy"]["accuracy"])
    return float(report["decord"][alpha]["accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-base", required=True)
    parser.add_argument("--dev-decord-sft", required=True)
    parser.add_argument("--dev-decord-ft", required=True)
    parser.add_argument("--test-base", required=True)
    parser.add_argument("--test-sft", required=True)
    parser.add_argument("--test-dpo", required=True)
    parser.add_argument("--test-decord-sft", required=True)
    parser.add_argument("--test-decord-ft", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--margin", type=float, default=0.01)
    args = parser.parse_args()
    dev_base = load(Path(args.dev_base))
    best_alpha = max(dev_base["decord"], key=lambda alpha: (dev_base["decord"][alpha]["accuracy"], -float(alpha)))
    dev_sft = load(Path(args.dev_decord_sft))
    dev_ft = load(Path(args.dev_decord_ft))
    use_dpo = dev_ft["greedy"]["accuracy"] >= dev_sft["greedy"]["accuracy"]
    ft_name = "decord_ft" if use_dpo else "decord_sft"
    test_base = load(Path(args.test_base))
    test_sft = load(Path(args.test_sft))
    test_dpo = load(Path(args.test_dpo))
    test_decord_sft = load(Path(args.test_decord_sft))
    test_decord_ft = load(Path(args.test_decord_ft))
    ft_report = test_decord_ft if use_dpo else test_decord_sft
    scores = {
        "base": accuracy(test_base, "greedy"),
        "sft": accuracy(test_sft, "greedy"),
        "dpo": accuracy(test_dpo, "greedy"),
        "decord": accuracy(test_base, "decord", best_alpha),
        "decord_ft": accuracy(ft_report, "greedy"),
    }
    comparisons = [scores["base"], scores["sft"], scores["dpo"]]
    bar = max(comparisons) + args.margin
    payload = {
        "alpha": best_alpha,
        "fine_tuned_variant": ft_name,
        "margin": args.margin,
        "test_accuracy": scores,
        "bar": bar,
        "decord_beats": scores["decord"] >= bar,
        "decord_ft_beats": scores["decord_ft"] >= bar,
        "goal_met": scores["decord"] >= bar and scores["decord_ft"] >= bar,
        "mismatch": {
            "base": test_base["greedy"]["mismatch_rate"],
            "sft": test_sft["greedy"]["mismatch_rate"],
            "dpo": test_dpo["greedy"]["mismatch_rate"],
            "decord": test_base["decord"][best_alpha]["mismatch_rate"],
            "decord_ft": ft_report["greedy"]["mismatch_rate"],
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
