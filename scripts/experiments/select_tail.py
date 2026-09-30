"""Choose alpha on the unused development tail.

Ties keep the smaller alpha. A short recommit continuation replaces that choice
only when its accuracy is strictly higher. This script does not read the test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tail-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.tail_json).read_text(encoding="utf-8"))
    phase2 = {alpha: float(summary["accuracy"]) for alpha, summary in report["decord"].items()}
    alpha = max(phase2, key=lambda item: (phase2[item], -float(item)))
    choice = {"rule": "phase2", "alpha": alpha, "tail_accuracy": phase2[alpha]}
    recommit = report.get("recommit")
    if recommit is not None and float(recommit["accuracy"]) > phase2[alpha]:
        choice = {"rule": "recommit", "alpha": None, "tail_accuracy": float(recommit["accuracy"])}
    choice["phase2"] = phase2
    if recommit is not None:
        choice["recommit_accuracy"] = float(recommit["accuracy"])
    Path(args.output_json).write_text(json.dumps(choice, indent=2), encoding="utf-8")
    print(json.dumps(choice, indent=2))


if __name__ == "__main__":
    main()
