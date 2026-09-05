#!/usr/bin/env python3
"""Merge per-model Stage A reports into reports/stage_a/stage_a_h1.json."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    models = []
    for path in sorted((root / "reports" / "stage_a").glob("*_stage_a_h1.json")):
        if path.name == "stage_a_h1.json":
            continue
        models.append(json.loads(path.read_text(encoding="utf-8")))
    if not models:
        raise SystemExit("no per-model stage_a reports")
    merged = {
        "n_models": len(models),
        "models": [
            {
                "model": m.get("model"),
                "primary_layer": m.get("primary_layer"),
                "n_eval": m.get("n_eval"),
                "behavior": m.get("behavior"),
                "h1_supported": m.get("h1_supported"),
                "control_artifact": m.get("control_artifact"),
                "primary_task_v": (m.get("layers") or {}).get(str(m.get("primary_layer")), {}).get("task_v"),
                "primary_random_v": (m.get("layers") or {}).get(str(m.get("primary_layer")), {}).get("random_v"),
                "shuffle_options": m.get("shuffle_options"),
            }
            for m in models
        ],
    }
    out = root / "reports" / "stage_a" / "stage_a_h1.json"
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(merged, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
