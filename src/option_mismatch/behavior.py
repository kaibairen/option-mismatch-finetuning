"""Letter / number / option-mismatch parsers for MCQ traces."""

from __future__ import annotations

import re
from typing import Any, Sequence

from option_mismatch.span import resolve_phase3_char_start

LETTER_LINE = re.compile(r"Final answer:\s*([A-E])", re.IGNORECASE)
LETTER_ANSWER_IS = re.compile(r"(?:答案是|answer\s+is|所以选)\s*[:：]?\s*([A-E])", re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+)")


def extract_letter(text: str) -> str:
    match = LETTER_LINE.search(text)
    if match:
        return match.group(1).upper()
    match = LETTER_ANSWER_IS.search(text)
    if match:
        return match.group(1).upper()
    letters = re.findall(r"\b([A-E])\b", text.upper())
    return letters[-1] if letters else ""


def parse_option_value(option: str) -> float | None:
    text = re.sub(r"^[A-Ea-e][\)\].:\s]+", "", str(option)).strip()
    text = text.replace(",", "").replace("%", "")
    match = NUMBER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def option_map(options: Sequence[str]) -> dict[str, float | None]:
    mapping: dict[str, float | None] = {}
    for opt in options:
        letter = str(opt).strip()[:1].upper()
        if letter not in "ABCDE":
            match = re.match(r"^\s*([A-E])", str(opt), re.IGNORECASE)
            letter = match.group(1).upper() if match else ""
        if letter:
            mapping[letter] = parse_option_value(opt)
    return mapping


def extract_numbers(text: str) -> list[float]:
    values = []
    for raw in NUMBER_RE.findall(text.replace(",", "")):
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return values


def numbers_close(a: float, b: float, *, rel: float = 1e-3, abs_tol: float = 1e-6) -> bool:
    return abs(a - b) <= max(abs_tol, rel * max(abs(a), abs(b), 1.0))


def infer_letter_from_numbers(numbers: list[float], options: Sequence[str]) -> str:
    mapping = option_map(options)
    hits: list[str] = []
    for value in reversed(numbers):
        for letter, opt_val in mapping.items():
            if opt_val is None:
                continue
            if numbers_close(value, opt_val):
                hits.append(letter)
        if hits:
            return hits[0]
    return ""


def analyze_behavior(generation: str, options: Sequence[str], gold: str) -> dict[str, Any]:
    gold_letter = str(gold or "").strip().upper()[:1]
    char_start, rule = resolve_phase3_char_start(generation, list(options))
    if rule == "not_found":
        cut = max(1, int(len(generation) * 0.70))
        phase2_text, phase3_text = generation[:cut], generation[cut:]
    else:
        phase2_text, phase3_text = generation[:char_start], generation[char_start:]

    pred = extract_letter(generation)
    format_ok = bool(LETTER_LINE.search(generation) or LETTER_ANSWER_IS.search(generation))
    phase2_numbers = extract_numbers(phase2_text)
    inferred = infer_letter_from_numbers(phase2_numbers, options)
    mapping = option_map(options)
    gold_value = mapping.get(gold_letter)
    number_match = False
    if gold_value is not None:
        number_match = any(numbers_close(v, gold_value) for v in phase2_numbers)
    if inferred and inferred == gold_letter:
        number_match = True

    letter_correct = pred == gold_letter and bool(pred)
    # Option mismatch: Phase II already points at an option, but the final letter flies.
    mismatch = False
    if inferred and pred and inferred != pred:
        mismatch = True
    elif (not letter_correct) and inferred == gold_letter and pred and pred != gold_letter:
        mismatch = True

    return {
        "pred": pred,
        "gold": gold_letter,
        "letter_correct": letter_correct,
        "format_ok": format_ok,
        "inferred_letter": inferred,
        "phase2_numbers": phase2_numbers[-8:],
        "number_match": number_match,
        "mismatch": mismatch,
        "phase3_rule": rule,
        "phase2_text": phase2_text,
        "phase3_text": phase3_text,
    }


def summarize_behavior(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "mismatch_rate": 0.0, "number_match_rate": 0.0, "format_rate": 0.0}

    def rate(key: str) -> float:
        return sum(1 for row in rows if row.get(key)) / n

    return {
        "n": n,
        "accuracy": rate("letter_correct"),
        "mismatch_rate": rate("mismatch"),
        "number_match_rate": rate("number_match"),
        "format_rate": rate("format_ok"),
    }
