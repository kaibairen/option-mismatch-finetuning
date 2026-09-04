from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def paired_drift_tests(phase2: np.ndarray, phase3: np.ndarray) -> dict[str, Any]:
    if phase2.shape != phase3.shape:
        raise ValueError("phase2 and phase3 must have the same shape")
    drift = phase2 - phase3
    t_stat, p_t = stats.ttest_rel(phase2, phase3, alternative="greater")
    try:
        w_stat, p_w = stats.wilcoxon(phase2, phase3, alternative="greater", zero_method="wilcox")
    except ValueError:
        w_stat, p_w = float("nan"), float("nan")
    std = float(np.std(drift, ddof=1)) if len(drift) > 1 else 0.0
    cohen_d = float(np.mean(drift) / std) if std > 0 else 0.0
    return {
        "n": int(len(drift)),
        "phase2_mean": float(np.mean(phase2)),
        "phase2_std": float(np.std(phase2, ddof=1)) if len(phase2) > 1 else 0.0,
        "phase3_mean": float(np.mean(phase3)),
        "phase3_std": float(np.std(phase3, ddof=1)) if len(phase3) > 1 else 0.0,
        "delta_ndi_drift": float(np.mean(drift)),
        "t_stat": float(t_stat),
        "p_ttest": float(p_t),
        "wilcoxon_stat": float(w_stat),
        "p_wilcoxon": float(p_w),
        "cohen_d": cohen_d,
        "h1_supported": bool(
            (p_t == p_t and p_w == p_w) and p_t < 0.005 and p_w < 0.005 and cohen_d > 0.3
        ),
    }
