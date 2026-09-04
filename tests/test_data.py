from pathlib import Path

from option_mismatch.io_utils import read_jsonl
from option_mismatch.prompts import format_mcq


def test_prepared_aqua_test_exists():
    path = Path("data/processed/aqua_test.jsonl")
    assert path.exists(), "run scripts/prepare_data.py first"
    rows = read_jsonl(path, limit=5)
    assert rows
    row = rows[0]
    assert row["question"]
    assert row["options"]
    assert row["correct"] in list("ABCDE")
    text = format_mcq(row["question"], row["options"])
    assert "Final answer" in text
