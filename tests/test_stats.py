from option_mismatch.stats import paired_drift_tests
import numpy as np


def test_h1_like_drift_is_significant():
    rng = np.random.default_rng(0)
    phase2 = rng.normal(0.1126, 0.0157, size=30)
    phase3 = phase2 - 0.0091 + rng.normal(0, 0.004, size=30)
    result = paired_drift_tests(phase2, phase3)
    assert result["n"] == 30
    assert result["delta_ndi_drift"] > 0
    assert result["p_ttest"] < 0.05
    assert result["cohen_d"] > 0


def test_no_drift():
    x = np.ones(20)
    result = paired_drift_tests(x, x)
    assert result["delta_ndi_drift"] == 0.0
    assert result["cohen_d"] == 0.0
