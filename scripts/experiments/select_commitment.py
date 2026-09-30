"""Freeze the commitment rule on a development report.

The full-trace number rule replaces the phase-2 rule only when its development
accuracy is strictly higher. Ties keep the phase-2 rule and the smaller alpha.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.dev_json).read_text(encoding="utf-8"))
    choices: dict[tuple[str, str], float] = {}
    for alpha, summary in report["decord"].items():
        choices[("phase2", alpha)] = float(summary["accuracy"])
    for alpha, summary in report.get("decord_fullnum", {}).items():
        choices[("fullnum", alpha)] = float(summary["accuracy"])
    kind, alpha = max(choices, key=lambda item: (choices[item], 1 if item[0] == "phase2" else 0, -float(item[1])))
    payload = {"rule": kind, "alpha": alpha, "dev_accuracy": choices[(kind, alpha)], "choices": {f"{k}:{a}": v for (k, a), v in choices.items()}}
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
