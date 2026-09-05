"""Content-based Phase II / Phase III cut. Token fraction is fallback only."""

from __future__ import annotations

import re
from typing import Any

PHASE3_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("final_answer", re.compile(r"final\s+answer\s*:", re.IGNORECASE)),
    ("answer_is_zh", re.compile(r"答案是")),
    ("so_choose_zh", re.compile(r"所以选")),
    ("the_answer_is", re.compile(r"the\s+answer\s+is", re.IGNORECASE)),
    ("therefore_choose", re.compile(r"therefore[, ]+the\s+answer", re.IGNORECASE)),
    ("choose_option", re.compile(r"(?:choose|select|pick)\s+(?:option\s+)?[A-E]\b", re.IGNORECASE)),
    ("standalone_letter", re.compile(r"(?m)^[ \t]*\(?([A-E])\)?[ \t]*$")),
]


def find_phase3_char_start(generation: str) -> tuple[int, str]:
    """Return (char_index, rule_name). char_index == len(generation) means not found."""
    best: tuple[int, str] | None = None
    for name, pattern in PHASE3_RULES:
        match = pattern.search(generation)
        if match is None:
            continue
        start = match.start()
        if best is None or start < best[0]:
            best = (start, name)
    if best is None:
        return len(generation), "not_found"
    return best


def option_alignment_char_start(generation: str, options: list[str] | None = None) -> tuple[int, str]:
    """First explicit mapping of a computed value onto an option letter."""
    pattern = re.compile(
        r"(?:option\s+)?([A-E])\s*(?:\)|:)?\s*(?:is|=|equals|对应|即为)",
        re.IGNORECASE,
    )
    match = pattern.search(generation)
    if match:
        return match.start(), "option_align"
    if options:
        for opt in options:
            letter = str(opt).strip()[:1].upper()
            if letter in "ABCDE" and re.search(rf"\b{letter}\b\s*(?:is|=)", generation, re.IGNORECASE):
                idx = re.search(rf"\b{letter}\b\s*(?:is|=)", generation, re.IGNORECASE)
                if idx:
                    return idx.start(), "option_align"
    return len(generation), "not_found"


def resolve_phase3_char_start(generation: str, options: list[str] | None = None) -> tuple[int, str]:
    start, rule = find_phase3_char_start(generation)
    if rule != "not_found":
        return start, rule
    return option_alignment_char_start(generation, options)


def char_to_token_index(tokenizer, text: str, char_index: int) -> int:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.get("offset_mapping")
    if not offsets:
        ids = encoded["input_ids"]
        if char_index >= len(text):
            return max(len(ids) - 1, 0)
        prefix = tokenizer(text[:char_index], add_special_tokens=False)["input_ids"]
        return min(len(prefix), max(len(ids) - 1, 0))
    if char_index >= len(text):
        return max(len(offsets) - 1, 0)
    for i, (lo, hi) in enumerate(offsets):
        if lo <= char_index < hi:
            return i
        if char_index < lo:
            return max(i - 1, 0)
    return max(len(offsets) - 1, 0)


def phase3_token_start(
    tokenizer,
    generation: str,
    *,
    options: list[str] | None = None,
    fallback_frac: float = 0.30,
) -> dict[str, Any]:
    n_tokens = len(tokenizer(generation, add_special_tokens=False)["input_ids"])
    char_start, rule = resolve_phase3_char_start(generation, options)
    used_fallback = rule == "not_found"
    if used_fallback:
        token_start = max(1, int(n_tokens * (1.0 - fallback_frac))) if n_tokens else 0
        rule = "token_frac_fallback"
    else:
        token_start = char_to_token_index(tokenizer, generation, char_start)
    token_start = min(max(token_start, 0), max(n_tokens - 1, 0))
    return {
        "char_start": char_start,
        "token_start": token_start,
        "n_tokens": n_tokens,
        "rule": rule,
        "used_fallback": used_fallback,
        "phase2_text": generation[:char_start] if not used_fallback else generation[: max(1, int(len(generation) * (1.0 - fallback_frac)))],
        "phase3_text": generation[char_start:] if not used_fallback else generation[max(1, int(len(generation) * (1.0 - fallback_frac))) :],
    }


def split_ndi(ndi, token_start: int) -> tuple[float, float]:
    import math

    if ndi is None or len(ndi) == 0:
        return float("nan"), float("nan")
    cut = min(max(int(token_start), 1), len(ndi))
    phase2 = ndi[:cut]
    phase3 = ndi[cut:] if cut < len(ndi) else ndi[-1:]
    if len(phase2) == 0 or len(phase3) == 0:
        return float("nan"), float("nan")
    p2 = float(sum(phase2) / len(phase2))
    p3 = float(sum(phase3) / len(phase3))
    if math.isnan(p2) or math.isnan(p3):
        return float("nan"), float("nan")
    return p2, p3
