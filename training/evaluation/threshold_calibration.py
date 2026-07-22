"""Per-pathology threshold calibration on the validation split.

The single rule that matters here: **thresholds are chosen on validation and
only ever read on test.** A threshold fitted on the test split is a hyperparameter
fitted on the test split, and every downstream number becomes optimistically
biased. :func:`calibrate_thresholds` therefore records the split it was fitted
on inside the output file, and :func:`load_thresholds` refuses a file whose
recorded split is a test split.

Objectives
----------
``f1``
    Maximise F1 of the positive class. The default: it balances precision and
    recall and matches the Stage-1 selection metric.
``youden_j``
    Maximise ``sensitivity + specificity - 1``. Insensitive to prevalence, so it
    behaves differently from F1 on rare findings.
``balanced_accuracy``
    Maximise ``(sensitivity + specificity) / 2``. A monotone transform of
    Youden's J, kept separate because it is what people report.
``recall_at_precision``
    Highest recall subject to ``precision >= constraint``. For a screening
    setting with a precision floor.
``precision_at_recall``
    Highest precision subject to ``recall >= constraint``. For a rule-out
    setting with a sensitivity floor.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from training.evaluation.schemas import ClassificationPredictions
from training.evaluation.uncertain_policy import (
    DEFAULT_POLICY,
    binarize_labels,
    validate_policy,
)

logger = logging.getLogger(__name__)

F1 = "f1"
YOUDEN_J = "youden_j"
BALANCED_ACCURACY = "balanced_accuracy"
RECALL_AT_PRECISION = "recall_at_precision"
PRECISION_AT_RECALL = "precision_at_recall"

OBJECTIVES = (F1, YOUDEN_J, BALANCED_ACCURACY, RECALL_AT_PRECISION, PRECISION_AT_RECALL)
DEFAULT_OBJECTIVE = F1
DEFAULT_THRESHOLD = 0.5

#: Splits a threshold file may legitimately have been fitted on.
CALIBRATION_SPLITS = ("validation", "val", "train")


class CalibrationError(ValueError):
    """Calibration was asked to do something statistically invalid."""


@dataclass
class PathologyThreshold:
    """The calibration outcome for one pathology."""

    name: str
    threshold: float
    objective_score: float
    default_threshold_score: float
    n_positive: int
    n_negative: int
    n_uncertain: int
    n_valid: int
    calibrated: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "objective_score": self.objective_score,
            "default_threshold_score": self.default_threshold_score,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_uncertain": self.n_uncertain,
            "n_valid": self.n_valid,
            "calibrated": self.calibrated,
            "reason": self.reason,
        }


@dataclass
class CalibrationResult:
    """All per-pathology thresholds plus the metadata needed to trust them."""

    thresholds: list[PathologyThreshold]
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_mapping(self) -> dict[str, float]:
        return {t.name: t.threshold for t in self.thresholds}

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "thresholds": {t.name: t.threshold for t in self.thresholds},
            "details": {t.name: t.to_dict() for t in self.thresholds},
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
        logger.info("wrote %d thresholds to %s", len(self.thresholds), path)
        return path


def _objective_score(
    objective: str,
    tp: np.ndarray,
    fp: np.ndarray,
    tn: np.ndarray,
    fn: np.ndarray,
    constraint: float,
) -> np.ndarray:
    """Vectorised objective value at every candidate threshold.

    Infeasible points under a constrained objective get ``-inf`` so they can
    never win the argmax.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
        recall = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
        specificity = np.where(tn + fp > 0, tn / np.maximum(tn + fp, 1), 0.0)

    if objective == F1:
        denominator = precision + recall
        return np.where(denominator > 0, 2 * precision * recall / np.maximum(denominator, 1e-12), 0.0)
    if objective == YOUDEN_J:
        return recall + specificity - 1.0
    if objective == BALANCED_ACCURACY:
        return (recall + specificity) / 2.0
    if objective == RECALL_AT_PRECISION:
        return np.where(precision >= constraint, recall, -np.inf)
    if objective == PRECISION_AT_RECALL:
        return np.where(recall >= constraint, precision, -np.inf)
    raise CalibrationError(
        f"unknown objective {objective!r}; expected one of {', '.join(OBJECTIVES)}"
    )


def calibrate_one(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    objective: str = DEFAULT_OBJECTIVE,
    constraint: float = 0.5,
) -> tuple[float, float]:
    """Return ``(threshold, objective_score)`` for one pathology.

    Candidate thresholds are the midpoints between consecutive distinct scores,
    plus the endpoints. Evaluating at observed scores directly would make the
    result depend on whether the comparison is ``>`` or ``>=``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true).astype(bool)
    if scores.size == 0:
        return DEFAULT_THRESHOLD, float("nan")

    distinct = np.unique(scores)
    if distinct.size == 1:
        return DEFAULT_THRESHOLD, float("nan")

    midpoints = (distinct[:-1] + distinct[1:]) / 2.0
    candidates = np.concatenate([[distinct[0] - 1e-6], midpoints, [distinct[-1] + 1e-6]])

    # predictions[i, j] = scores[j] >= candidates[i]
    predictions = scores[None, :] >= candidates[:, None]
    positives = y_true[None, :]

    tp = np.sum(predictions & positives, axis=1).astype(np.float64)
    fp = np.sum(predictions & ~positives, axis=1).astype(np.float64)
    tn = np.sum(~predictions & ~positives, axis=1).astype(np.float64)
    fn = np.sum(~predictions & positives, axis=1).astype(np.float64)

    values = _objective_score(objective, tp, fp, tn, fn, constraint)
    if not np.any(np.isfinite(values)):
        return DEFAULT_THRESHOLD, float("nan")

    best = int(np.nanargmax(np.where(np.isfinite(values), values, -np.inf)))
    return float(candidates[best]), float(values[best])


def calibrate_thresholds(
    predictions: ClassificationPredictions,
    *,
    split: str,
    objective: str = DEFAULT_OBJECTIVE,
    uncertain_policy: str = DEFAULT_POLICY,
    constraint: float = 0.5,
    min_positive: int = 1,
) -> CalibrationResult:
    """Fit one threshold per pathology on a validation-like split.

    Raises
    ------
    CalibrationError
        If ``split`` names a test split. Calibrating on test is the failure
        this function is designed to make impossible rather than merely
        discouraged.
    """
    if objective not in OBJECTIVES:
        raise CalibrationError(
            f"unknown objective {objective!r}; expected one of {', '.join(OBJECTIVES)}"
        )
    validate_policy(uncertain_policy)

    normalised = split.strip().lower()
    if normalised not in CALIBRATION_SPLITS:
        raise CalibrationError(
            f"refusing to calibrate thresholds on split {split!r}. Thresholds "
            "fitted on a test split make every test metric optimistically "
            f"biased. Expected one of: {', '.join(CALIBRATION_SPLITS)}."
        )

    y_true, valid = binarize_labels(predictions.labels, uncertain_policy)
    positive_probabilities = predictions.positive_probabilities

    results: list[PathologyThreshold] = []
    for index, name in enumerate(predictions.pathology_names):
        column_valid = valid[:, index]
        scores = positive_probabilities[column_valid, index]
        truth = y_true[column_valid, index]

        raw_column = predictions.labels[:, index]
        n_positive = int(np.sum(truth == 1))
        n_negative = int(np.sum(truth == 0))
        n_uncertain = int(np.sum(raw_column == 2))

        if n_positive < min_positive or n_negative == 0:
            reason = (
                f"only {n_positive} positive and {n_negative} negative samples; "
                f"need at least {min_positive} positive and 1 negative"
            )
            logger.warning("%s: not calibrated (%s)", name, reason)
            results.append(
                PathologyThreshold(
                    name=name,
                    threshold=DEFAULT_THRESHOLD,
                    objective_score=float("nan"),
                    default_threshold_score=float("nan"),
                    n_positive=n_positive,
                    n_negative=n_negative,
                    n_uncertain=n_uncertain,
                    n_valid=int(truth.size),
                    calibrated=False,
                    reason=reason,
                )
            )
            continue

        threshold, score = calibrate_one(
            scores, truth, objective=objective, constraint=constraint
        )
        baseline_score = _score_at(
            scores, truth, DEFAULT_THRESHOLD, objective, constraint
        )
        results.append(
            PathologyThreshold(
                name=name,
                threshold=threshold,
                objective_score=score,
                default_threshold_score=baseline_score,
                n_positive=n_positive,
                n_negative=n_negative,
                n_uncertain=n_uncertain,
                n_valid=int(truth.size),
                calibrated=True,
            )
        )

    metadata = {
        "split": split,
        "objective": objective,
        "uncertain_policy": uncertain_policy,
        "constraint": constraint,
        "default_threshold": DEFAULT_THRESHOLD,
        "num_samples": predictions.num_samples,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_metadata": predictions.metadata,
    }
    return CalibrationResult(thresholds=results, metadata=metadata)


def _score_at(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    objective: str,
    constraint: float,
) -> float:
    predicted = scores >= threshold
    truth = np.asarray(y_true).astype(bool)
    tp = np.asarray([np.sum(predicted & truth)], dtype=np.float64)
    fp = np.asarray([np.sum(predicted & ~truth)], dtype=np.float64)
    tn = np.asarray([np.sum(~predicted & ~truth)], dtype=np.float64)
    fn = np.asarray([np.sum(~predicted & truth)], dtype=np.float64)
    value = _objective_score(objective, tp, fp, tn, fn, constraint)[0]
    return float(value) if np.isfinite(value) else float("nan")


def load_thresholds(path: str | Path, *, allow_test_split: bool = False) -> dict[str, float]:
    """Read a threshold file, refusing one that was fitted on test.

    ``allow_test_split`` exists only so a test can prove the guard fires; there
    is no legitimate reason to set it in a real evaluation.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"threshold file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if "thresholds" not in payload:
        raise CalibrationError(f"{path} has no 'thresholds' key")

    split = str(payload.get("metadata", {}).get("split", "")).strip().lower()
    if not allow_test_split and split and split not in CALIBRATION_SPLITS:
        raise CalibrationError(
            f"{path} records split={split!r}. Thresholds must be calibrated on "
            "validation; using test-fitted thresholds would bias every test "
            "metric. Re-run calibration against the validation split."
        )
    if not split:
        logger.warning(
            "%s records no split in its metadata; cannot verify it was "
            "calibrated on validation",
            path,
        )

    return {str(k): float(v) for k, v in payload["thresholds"].items()}
