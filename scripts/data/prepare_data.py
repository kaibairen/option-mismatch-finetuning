#!/usr/bin/env python3
"""Download AQUA-RAT from DeepMind and emit jsonl splits used by the probe."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from urllib.request import urlopen

AQUA_RAW = {
    "train": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/train.json",
    "dev": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/dev.json",
    "test": "https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json",
}


def load_json_records(source: str | Path) -> list[dict]:
    if str(source).startswith("http"):
        with urlopen(source, timeout=180) as resp:
            text = resp.read().decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def to_record(row: dict) -> dict:
    options = row.get("options") or []
    return {
        "question": row.get("question", "").strip(),
        "options": list(options),
        "rationale": (row.get("rationale") or "").strip(),
        "correct": str(row.get("correct") or "").strip().upper()[:1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=sorted(AQUA_RAW))
    parser.add_argument("--limit", type=int, default=0, help="0 keeps the full split (applied after sampling)")
    parser.add_argument("--sample", type=int, default=0, help="random sample size; 0 keeps all rows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-name", default="", help="override jsonl stem, e.g. aqua_test_100")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    raw_dir = args.root / "data" / "raw"
    out_dir = args.root / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    url = AQUA_RAW[args.split]
    raw_path = raw_dir / f"aqua_{args.split}.json"
    if raw_path.exists() and raw_path.stat().st_size > 0:
        rows = load_json_records(raw_path)
        print(f"[data] cache hit {raw_path} ({len(rows)} rows)")
    else:
        print(f"[data] downloading {url}")
        rows = load_json_records(url)
        raw_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    records = [to_record(row) for row in rows if row.get("question")]
    if args.sample > 0 and args.sample < len(records):
        rng = random.Random(args.seed)
        records = rng.sample(records, args.sample)
    if args.limit > 0:
        records = records[: args.limit]

    stem = args.output_name or f"aqua_{args.split}"
    out_path = out_dir / f"{stem}.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[data] wrote {len(records)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
