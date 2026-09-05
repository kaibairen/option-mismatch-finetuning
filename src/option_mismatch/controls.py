"""Negative controls: random diligence vectors and shuffled options."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np


def random_unit_vector(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim).astype(np.float64)
    norm = np.linalg.norm(vec)
    if norm == 0:
        vec[0] = 1.0
        return vec
    return vec / norm


def permute_vector(vec: np.ndarray, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.array(vec, copy=True)
    rng.shuffle(out)
    norm = np.linalg.norm(out)
    return out / norm if norm else out


def shuffle_options(options: Sequence[str], seed: int) -> list[str]:
    """Keep A-E prefixes, shuffle the payloads so the mapping is destroyed."""
    payloads = []
    for opt in options:
        text = str(opt)
        if len(text) >= 2 and text[0].upper() in "ABCDE" and text[1] in ")]:. ":
            payloads.append(text[2:].lstrip())
        else:
            payloads.append(text)
    rng = random.Random(seed)
    rng.shuffle(payloads)
    letters = "ABCDE"
    rebuilt = []
    for i, payload in enumerate(payloads):
        letter = letters[i] if i < len(letters) else chr(ord("A") + i)
        rebuilt.append(f"{letter}){payload}")
    return rebuilt


def zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    if std == 0:
        return arr - float(np.mean(arr))
    return (arr - float(np.mean(arr))) / std
