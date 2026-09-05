#!/usr/bin/env python3
"""Compare Base / SFT / DPO / Rep-DPO against the locked success bar."""

from __future__ import annotations

import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("summary") or payload


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    names = [("base", "eval_base.json"), ("sft", "eval_sft.json"), ("dpo", "eval_dpo.json"), ("rep_dpo", "eval_rep_dpo.json")]
    methods = {}
    for name, filename in names:
        path = root / "results" / filename
        if path.exists():
            methods[name] = load_summary(path)
    if "base" not in methods or "rep_dpo" not in methods:
        raise SystemExit(f"missing eval jsons in {root / 'results'}: {list(methods)}")
    base_m = methods["base"].get("mismatch_rate", 1.0)
    rep_m = methods["rep_dpo"].get("mismatch_rate", 1.0)
    dpo_m = methods.get("dpo", {}).get("mismatch_rate", base_m)
    sft_m = methods.get("sft", {}).get("mismatch_rate", base_m)
    base_a = methods["base"].get("accuracy", 0.0)
    rep_a = methods["rep_dpo"].get("accuracy", 0.0)
    base_p3 = ((methods["base"].get("ndi") or {}).get("phase3_mean"))
    rep_p3 = ((methods["rep_dpo"].get("ndi") or {}).get("phase3_mean"))
    mismatch_drop_pp = (base_m - rep_m) * 100
    acc_lift_pp = (rep_a - base_a) * 100
    better_than_baselines = rep_m < dpo_m and rep_m < sft_m
    ndi_up = (rep_p3 is not None and base_p3 is not None and rep_p3 > base_p3)
    primary = mismatch_drop_pp >= 5 and better_than_baselines and ndi_up
    strong = primary and acc_lift_pp >= 3
    report = {
        "methods": methods,
        "mismatch_drop_pp": mismatch_drop_pp,
        "accuracy_lift_pp": acc_lift_pp,
        "better_than_sft_dpo": better_than_baselines,
        "phase3_ndi_up": ndi_up,
        "primary_success": primary,
        "strong_success": strong,
    }
    out = root / "reports" / "stage_b" / "stage_b_repair.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "methods"}, indent=2))
    print(json.dumps({k: methods[k] for k in methods}, indent=2))


if __name__ == "__main__":
    main()
