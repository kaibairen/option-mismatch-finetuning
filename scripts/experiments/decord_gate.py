"""Apply the dev-selected DECORD alpha and check the +1 percentage-point gate on test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy(report: dict, key: str) -> float:
    return float(report["greedy"]["accuracy"])


def decord_choices(report: dict) -> dict[tuple[str, str], float]:
    choices = {}
    for alpha, summary in report.get("decord", {}).items():
        choices[("decord", alpha)] = float(summary["accuracy"])
    for alpha, summary in report.get("decord_majority", {}).items():
        choices[("majority", alpha)] = float(summary["accuracy"])
    return choices


def decord_accuracy(report: dict, kind: str, alpha: str) -> float:
    block = report["decord"] if kind == "decord" else report["decord_majority"]
    return float(block[alpha]["accuracy"])


def decord_mismatch(report: dict, kind: str, alpha: str) -> float:
    block = report["decord"] if kind == "decord" else report["decord_majority"]
    return float(block[alpha]["mismatch_rate"])


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
    choices = decord_choices(dev_base)
    # Equal dev scores keep the single-trace rule and the smaller alpha.
    # Four-sample voting did not beat that rule on the development slice.
    kind, best_alpha = max(choices, key=lambda item: (choices[item], 1 if item[0] == "decord" else 0, -float(item[1])))
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
        "decord": decord_accuracy(test_base, kind, best_alpha),
        "decord_ft": accuracy(ft_report, "greedy"),
    }
    comparisons = [scores["base"], scores["sft"], scores["dpo"]]
    bar = max(comparisons) + args.margin
    payload = {
        "alpha": best_alpha,
        "decord_kind": kind,
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
            "decord": decord_mismatch(test_base, kind, best_alpha),
            "decord_ft": ft_report["greedy"]["mismatch_rate"],
        },
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
