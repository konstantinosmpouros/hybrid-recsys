import numpy as np
import pytest
from hybrid_recsys.evaluation.metrics import (
    rmse, mae, precision_at_k, recall_at_k, f1_at_k,
    evaluate_rating_prediction,
)


# ── rmse / mae ────────────────────────────────────────────────────────────────

def test_rmse_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_known_value():
    true = np.array([3.0, 4.0])
    pred = np.array([4.0, 3.0])
    assert rmse(true, pred) == pytest.approx(1.0)


def test_rmse_ignores_nan():
    true = np.array([3.0, 4.0, 5.0])
    pred = np.array([3.0, np.nan, 5.0])
    assert rmse(true, pred) == pytest.approx(0.0)


def test_mae_known_value():
    true = np.array([3.0, 5.0])
    pred = np.array([4.0, 4.0])
    assert mae(true, pred) == pytest.approx(1.0)


def test_evaluate_rating_prediction_returns_both_keys():
    result = evaluate_rating_prediction(np.array([4.0, 3.0]), np.array([3.0, 4.0]))
    assert "rmse" in result
    assert "mae" in result


# ── precision / recall / f1 ───────────────────────────────────────────────────

def test_precision_at_k_perfect():
    assert precision_at_k([1, 2, 3], {1, 2, 3}) == pytest.approx(1.0)


def test_precision_at_k_zero():
    assert precision_at_k([1, 2, 3], {4, 5, 6}) == pytest.approx(0.0)


def test_precision_at_k_partial():
    assert precision_at_k([1, 2, 3, 4], {1, 3}) == pytest.approx(0.5)


def test_precision_at_k_empty_recommended():
    assert precision_at_k([], {1, 2}) == pytest.approx(0.0)


def test_recall_at_k_perfect():
    assert recall_at_k([1, 2, 3], {1, 2, 3}) == pytest.approx(1.0)


def test_recall_at_k_partial():
    assert recall_at_k([1, 2], {1, 2, 3}) == pytest.approx(2 / 3)


def test_recall_at_k_empty_relevant():
    assert recall_at_k([1, 2], set()) == pytest.approx(0.0)


def test_f1_at_k_zero_both():
    assert f1_at_k(0.0, 0.0) == pytest.approx(0.0)


def test_f1_at_k_balanced():
    assert f1_at_k(0.5, 0.5) == pytest.approx(0.5)


def test_f1_at_k_harmonic_mean():
    p, r = 0.4, 0.6
    expected = 2 * p * r / (p + r)
    assert f1_at_k(p, r) == pytest.approx(expected)
