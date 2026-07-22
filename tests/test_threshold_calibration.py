"""Threshold calibration, bootstrap and baseline tests."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.test_classification_metrics import make_predictions  # noqa: E402
from training.evaluation.baselines import (  # noqa: E402
    ALL_NEGATIVE,
    compute_baselines,
)
from training.evaluation.bootstrap import (  # noqa: E402
    BootstrapError,
    bootstrap_metric,
    bootstrap_sample_metric,
)
from training.evaluation.classification_metrics import (  # noqa: E402
    evaluate_classification,
)
from training.evaluation.threshold_calibration import (  # noqa: E402
    BALANCED_ACCURACY,
    F1,
    PRECISION_AT_RECALL,
    RECALL_AT_PRECISION,
    YOUDEN_J,
    CalibrationError,
    calibrate_one,
    calibrate_thresholds,
    load_thresholds,
)

# --------------------------------------------------------------------------
# Calibration must never touch test data
# --------------------------------------------------------------------------


def test_calibrating_on_test_split_is_refused():
    preds = make_predictions(np.array([[1], [0]]), np.array([[0.9], [0.1]]))
    with pytest.raises(CalibrationError, match="refusing to calibrate"):
        calibrate_thresholds(preds, split="test")


def test_loading_test_fitted_thresholds_is_refused(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"metadata": {"split": "test"}, "thresholds": {"P0": 0.3}})
    )
    with pytest.raises(CalibrationError, match="calibrated on validation"):
        load_thresholds(path)
    # The guard is the only thing standing between the two; prove it is what
    # fires, not a parse error.
    assert load_thresholds(path, allow_test_split=True) == {"P0": 0.3}


def test_validation_thresholds_load_cleanly(tmp_path):
    preds = make_predictions(
        np.array([[1], [1], [0], [0]]), np.array([[0.8], [0.7], [0.3], [0.2]])
    )
    result = calibrate_thresholds(preds, split="validation")
    path = result.save(tmp_path / "thresholds.json")
    assert load_thresholds(path) == result.as_mapping()


# --------------------------------------------------------------------------
# The threshold actually moves, and moves in the right direction
# --------------------------------------------------------------------------


def test_calibrated_threshold_differs_from_half_when_scores_are_shifted():
    """All scores sit below 0.5, so 0.5 detects nothing.

    Calibration must find a cutoff inside the score range instead.
    """
    scores = np.array([0.40, 0.35, 0.20, 0.10])
    truth = np.array([1, 1, 0, 0])
    threshold, score = calibrate_one(scores, truth, objective=F1)
    assert 0.20 < threshold < 0.35
    assert score == pytest.approx(1.0)
    assert threshold != 0.5


def test_calibration_improves_f1_over_default_threshold():
    labels = np.array([[1], [1], [0], [0], [0], [0]])
    positive = np.array([[0.45], [0.40], [0.30], [0.20], [0.10], [0.05]])
    preds = make_predictions(labels, positive)

    at_half = evaluate_classification(preds)
    assert at_half.aggregates["positive_macro_f1"] == 0.0  # nothing crosses 0.5

    result = calibrate_thresholds(preds, split="validation", objective=F1)
    calibrated = evaluate_classification(preds, thresholds=result.as_mapping())
    assert calibrated.aggregates["positive_macro_f1"] == pytest.approx(1.0)

    detail = result.thresholds[0]
    assert detail.calibrated
    assert detail.objective_score > detail.default_threshold_score


def test_every_objective_runs_and_records_its_score():
    scores = np.array([0.9, 0.7, 0.6, 0.4, 0.3, 0.1])
    truth = np.array([1, 1, 0, 1, 0, 0])
    for objective in (F1, YOUDEN_J, BALANCED_ACCURACY):
        threshold, score = calibrate_one(scores, truth, objective=objective)
        assert 0.0 <= threshold <= 1.0
        assert not math.isnan(score)


def test_constrained_objectives_respect_the_constraint():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    truth = np.array([1, 0, 1, 0, 1, 0])

    # Demand precision >= 1.0: only the top-1 cutoff qualifies, recall = 1/3.
    threshold, recall = calibrate_one(
        scores, truth, objective=RECALL_AT_PRECISION, constraint=1.0
    )
    assert recall == pytest.approx(1 / 3)

    # Demand recall >= 1.0: the cutoff must reach the lowest positive.
    threshold, precision = calibrate_one(
        scores, truth, objective=PRECISION_AT_RECALL, constraint=1.0
    )
    assert threshold < 0.5
    assert precision == pytest.approx(0.6)


def test_unsatisfiable_constraint_falls_back_to_default():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    truth = np.array([0, 0, 0, 1])
    # Precision 1.0 is unreachable except at a cutoff that also has recall 0.
    threshold, score = calibrate_one(
        scores, truth, objective=RECALL_AT_PRECISION, constraint=1.0
    )
    assert threshold <= 1.0  # a value is still returned; nothing crashes


def test_pathology_without_positives_keeps_default_threshold():
    labels = np.array([[0], [0], [0]])
    preds = make_predictions(labels, np.array([[0.9], [0.5], [0.1]]))
    result = calibrate_thresholds(preds, split="validation")
    detail = result.thresholds[0]
    assert detail.threshold == 0.5
    assert not detail.calibrated
    assert "positive" in detail.reason


def test_calibration_records_class_counts_and_metadata():
    labels = np.array([[1], [2], [0], [-1]])
    preds = make_predictions(labels, np.array([[0.9], [0.6], [0.2], [0.1]]))
    result = calibrate_thresholds(preds, split="validation", objective=F1)

    assert result.metadata["split"] == "validation"
    assert result.metadata["objective"] == F1
    detail = result.thresholds[0]
    assert detail.n_positive == 1
    assert detail.n_uncertain == 1
    assert detail.n_valid == 3  # the -1 row is excluded


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


def test_bootstrap_is_reproducible_with_the_same_seed():
    values = np.linspace(0, 1, 50)

    def mean_of(indices: np.ndarray) -> float:
        return float(np.mean(values[indices]))

    first = bootstrap_metric(mean_of, 50, samples=100, seed=7)
    second = bootstrap_metric(mean_of, 50, samples=100, seed=7)
    assert (first.lower, first.upper) == (second.lower, second.upper)

    different = bootstrap_metric(mean_of, 50, samples=100, seed=8)
    assert (different.lower, different.upper) != (first.lower, first.upper)


def test_bootstrap_interval_brackets_the_point_estimate():
    values = np.random.default_rng(0).normal(0.5, 0.1, size=200)
    interval = bootstrap_sample_metric(values, samples=200, seed=1)
    assert interval.lower <= interval.point_estimate <= interval.upper
    assert interval.valid_replicates == 200


def test_bootstrap_can_be_disabled():
    interval = bootstrap_sample_metric([0.1, 0.2, 0.3], samples=0)
    assert interval.method == "disabled"
    assert math.isnan(interval.lower)
    assert interval.point_estimate == pytest.approx(0.2)


def test_bootstrap_drops_undefined_replicates_instead_of_scoring_them_zero():
    def sometimes_undefined(indices: np.ndarray) -> float:
        return float("nan") if indices[0] % 2 == 0 else 1.0

    interval = bootstrap_metric(sometimes_undefined, 10, samples=50, seed=3)
    assert interval.valid_replicates < 50
    assert interval.lower == pytest.approx(1.0)


def test_bootstrap_rejects_invalid_configuration():
    with pytest.raises(BootstrapError):
        bootstrap_metric(lambda i: 1.0, 0, samples=10)
    with pytest.raises(BootstrapError):
        bootstrap_metric(lambda i: 1.0, 5, samples=10, confidence=1.5)


def test_confidence_interval_json_has_no_nan():
    interval = bootstrap_sample_metric([float("nan")], samples=0)
    payload = interval.to_dict()
    assert payload["lower"] is None
    # json.dumps with allow_nan=False would raise on a real NaN.
    json.dumps(payload, allow_nan=False)


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------


def test_all_negative_baseline_exposes_deceptive_accuracy():
    """19/20 negative: the baseline gets 95% accuracy and 0 positive F1."""
    labels = np.zeros((10, 2), dtype=int)
    labels[0, 0] = 1
    labels[0, 1] = 1
    preds = make_predictions(labels, np.full((10, 2), 0.9))

    rows = {row.name: row for row in compute_baselines(preds)}
    negative = rows[ALL_NEGATIVE]
    assert negative.accuracy == pytest.approx(0.9)
    assert negative.positive_macro_f1 == 0.0


def test_baselines_cover_every_requested_variant():
    labels = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
    preds = make_predictions(labels, np.array([[0.9, 0.1]] * 4))
    rows = compute_baselines(preds, seed=0)
    assert {row.name for row in rows} == {
        "all_negative",
        "all_positive",
        "majority_class",
        "prevalence_random",
        "threshold_half",
    }
    for row in rows:
        assert row.description
