"""DECORD: Diligence-guided End Commitment for Option Representation Drift.

Training-free letter selection. After a solution trace, each option letter is
scored by the model's next-token log-probability at a forced ``Final answer:``
boundary, plus a bonus when that letter is the one implied by the numbers
computed before the boundary.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from option_mismatch.behavior import analyze_behavior, extract_letter
from option_mismatch.span import resolve_phase3_char_start


def blend_letter_scores(
    letter_logprobs: Mapping[str, float],
    inferred: str,
    alpha: float,
) -> dict[str, float]:
    """Add ``alpha`` nats to the inferred letter and subtract ``alpha`` from the others.

    ``alpha == 0`` is pure next-token choice. A non-empty inferred letter with
    ``alpha > 0`` pulls the decision toward the number the trace already computed.
    """
    scores = {str(letter): float(value) for letter, value in letter_logprobs.items()}
    inferred = (inferred or "").strip().upper()[:1]
    if not inferred or alpha == 0:
        return scores
    for letter in list(scores):
        scores[letter] = scores[letter] + (float(alpha) if letter == inferred else -float(alpha))
    return scores


def argmax_letter(scores: Mapping[str, float]) -> str:
    if not scores:
        return ""
    return max(scores, key=lambda letter: (scores[letter], letter))


def phase2_prefix(generation: str, options: list[str] | None = None) -> str:
    start, rule = resolve_phase3_char_start(generation, options)
    if rule == "not_found":
        return generation.rstrip()
    return generation[:start].rstrip()


def select_decord(traces: list[dict[str, Any]], *, alpha: float) -> dict[str, Any]:
    """Pick one letter from scored traces. Each trace may carry ``letter_logprobs``."""
    best: dict[str, Any] | None = None
    for trace in traces:
        logprobs = trace.get("letter_logprobs") or {}
        inferred = str(trace.get("inferred_letter") or "")
        scores = blend_letter_scores(logprobs, inferred, alpha)
        if scores:
            letter = argmax_letter(scores)
            value = float(scores[letter])
        else:
            letter = str(trace.get("pred") or "")
            value = 1.0 if letter and letter == inferred else 0.0
        row = {
            "letter": letter,
            "score": value,
            "pred": trace.get("pred") or "",
            "inferred_letter": inferred,
            "generation": trace.get("generation") or "",
        }
        if best is None or row["score"] > best["score"]:
            best = row
    if best is None:
        return {"letter": "", "score": float("-inf"), "pred": "", "inferred_letter": "", "generation": ""}
    return best


def letter_token_candidates(tokenizer, letter: str) -> list[int]:
    found: list[int] = []
    for variant in (f" {letter}", letter):
        ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(ids) == 1:
            found.append(int(ids[0]))
    if not found:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            found.append(int(ids[-1]))
    return list(dict.fromkeys(found))


def letter_token_ids(tokenizer) -> dict[str, int]:
    """One id per letter, preferring the leading-space form used after a colon."""
    return {letter: letter_token_candidates(tokenizer, letter)[0] for letter in "ABCDE"}


@torch.no_grad()
def letter_logprobs(model, tokenizer, prefix: str) -> dict[str, float]:
    device = next(model.parameters()).device
    encoded = tokenizer(prefix, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    logits = model(**encoded, use_cache=False).logits[0, -1].float()
    logp = torch.log_softmax(logits, dim=-1)
    scores: dict[str, float] = {}
    for letter in "ABCDE":
        ids = letter_token_candidates(tokenizer, letter)
        scores[letter] = max(float(logp[token_id].item()) for token_id in ids)
    return scores


@torch.no_grad()
def decord_from_generation(
    model,
    tokenizer,
    prompt: str,
    generation: str,
    options: list[str],
    gold: str,
    *,
    alpha: float,
) -> dict[str, Any]:
    """Rescore one finished trace. ``gold`` is used only to fill behavior fields."""
    behavior = analyze_behavior(generation, options, gold)
    prefix_body = phase2_prefix(generation, options)
    prefix = f"{prompt}{prefix_body}\nFinal answer:"
    scores = letter_logprobs(model, tokenizer, prefix)
    chosen = select_decord(
        [
            {
                "letter_logprobs": scores,
                "inferred_letter": behavior.get("inferred_letter") or "",
                "pred": behavior.get("pred") or "",
                "generation": generation,
            }
        ],
        alpha=alpha,
    )
    letter = chosen["letter"] or extract_letter(generation)
    return {
        "letter": letter,
        "score": chosen["score"],
        "letter_logprobs": scores,
        "behavior": behavior,
        "alpha": alpha,
    }
