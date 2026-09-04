import numpy as np

from option_mismatch.controls import permute_vector, random_unit_vector, shuffle_options, zscore
from option_mismatch.stats import paired_drift_tests


def test_random_vector_is_unit():
    v = random_unit_vector(16, seed=3)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9


def test_shuffle_options_keeps_letters_moves_payloads():
    options = ["A)125", "B)150", "C)225"]
    shuffled = shuffle_options(options, seed=0)
    assert [row[0] for row in shuffled] == ["A", "B", "C"]
    assert {row.split(")", 1)[1] for row in shuffled} == {"125", "150", "225"}


def test_random_projection_delta_near_zero():
    rng = np.random.default_rng(1)
    hidden = rng.normal(size=(40, 8))
    hidden = hidden / np.linalg.norm(hidden, axis=1, keepdims=True)
    v = random_unit_vector(8, seed=9)
    scores = hidden @ v
    phase2 = scores[:20]
    phase3 = scores[20:]
    # random split should not systematically prefer phase2
    result = paired_drift_tests(phase2, phase3)
    assert result["p_ttest"] > 0.01 or abs(result["cohen_d"]) < 0.5


def test_zscore_centers():
    z = zscore(np.array([1.0, 2.0, 3.0]))
    assert abs(float(np.mean(z))) < 1e-9
