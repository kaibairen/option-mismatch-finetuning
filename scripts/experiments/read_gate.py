"""Print accuracies from finished DECORD json reports and the +1pp gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_decord(report: dict) -> tuple[str, float]:
    choices: dict[str, float] = {}
    for alpha, summary in report.get("decord", {}).items():
        choices[f"decord@{alpha}"] = float(summary["accuracy"])
    for alpha, summary in report.get("decord_majority", {}).items():
        choices[f"majority@{alpha}"] = float(summary["accuracy"])
    if not choices:
        return "none", float("nan")
    name = max(choices, key=lambda key: (choices[key], key))
    return name, choices[name]


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/decord")
    names = [
        "dev_base",
        "dev_decord_sft",
        "dev_decord_ft",
        "test_base",
        "test_sft",
        "test_dpo",
        "test_decord_sft",
        "test_decord_ft",
    ]
    reports = {name: load(root / f"{name}.json") for name in names}
    for name, report in reports.items():
        if report is None:
            print(f"{name}: missing")
            continue
        kind, score = best_decord(report)
        print(
            f"{name}: greedy={report['greedy']['accuracy']:.4f} "
            f"mismatch={report['greedy']['mismatch_rate']:.4f} "
            f"format={report['greedy']['format_rate']:.4f} "
            f"best_{kind}={score:.4f}"
        )
    needed = ["test_base", "test_sft", "test_dpo", "test_decord_sft", "test_decord_ft"]
    if any(reports[name] is None for name in needed):
        return
    base = reports["test_base"]
    sft = reports["test_sft"]
    dpo = reports["test_dpo"]
    ft_sft = reports["test_decord_sft"]
    ft = reports["test_decord_ft"]
    bar = max(base["greedy"]["accuracy"], sft["greedy"]["accuracy"], dpo["greedy"]["accuracy"]) + 0.01
    _, base_decord = best_decord(base)
    ft_greedy = max(ft_sft["greedy"]["accuracy"], ft["greedy"]["accuracy"])
    _, ft_decord = best_decord(ft if ft["greedy"]["accuracy"] >= ft_sft["greedy"]["accuracy"] else ft_sft)
    print(f"bar={bar:.4f}")
    print(f"base_decord_beats={base_decord >= bar} ft_greedy_beats={ft_greedy >= bar} ft_decord_beats={ft_decord >= bar}")


if __name__ == "__main__":
    main()
