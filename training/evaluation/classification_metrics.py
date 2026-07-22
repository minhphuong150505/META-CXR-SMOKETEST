"""Stage-1 classification metrics.

Implemented in **numpy only**, deliberately. scikit-learn is not installed in
the CPU development environment and pulling it in would make the metric core
untestable there. Every formula below is standard and is unit-tested against
hand-computed values in ``tests/test_classification_metrics.py``.

The defect this module exists to fix
------------------------------------
``model/lavis/tasks/image_text_pretrain.py`` computes

    recall = true_positive / support.clamp_min(1.0)
    f1_positive_macro = f1[:, 1].mean()

so a pathology with **no positive samples in the split** contributes a hard 0 to
the macro average instead of being excluded. The macro score therefore moves
with the split's class composition, which makes validation and test numbers
incomparable -- and it is the checkpoint selection metric.

Here, an undefined per-pathology value is ``nan``, macro aggregates are
``nanmean`` over defined entries only, and every skipped pathology is reported
by name with the reason. Nothing is silently averaged as zero.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from training.evaluation.schemas import (
    CLASS_NAMES,
    ClassificationPredictions,
    SchemaError,
)
from training.evaluation.uncertain_policy import (
    DEFAULT_POLICY,
    binarize_labels,
    describe_policy,
    validate_policy,
)

logger = logging.getLogger(__name__)

#: Reasons a per-pathology metric can be undefined.
NO_POSITIVE_SAMPLES = "no_positive_samples"
NO_NEGATIVE_SAMPLES = "no_negative_samples"
NO_VALID_SAMPLES = "no_valid_samples"
NEVER_PREDICTED_POSITIVE = "never_predicted_positive"


def _safe_divide(numerator: float, denominator: float) -> float:
    """Return ``nan`` when the denominator is zero, never 0.0.

    This is the whole difference between "the model got none of them right" and
    "there was nothing to get right". Collapsing the second into the first is
    the bug this module was written to remove.
    """
    if denominator == 0:
        return float("nan")
    return float(numerator) / float(denominator)


def binary_confusion(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)`` for one pathology."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    return tp, fp, tn, fn


def roc_auc(scores: np.ndarray, y_true: np.ndarray) -> float:
    """AUROC via the rank (Mann-Whitney U) identity, with tie correction.

    Returns ``nan`` when either class is absent -- AUROC is undefined then, and
    returning 0.5 would look like a real measurement of a coin flip.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true).astype(bool)
    n_pos = int(np.sum(y_true))
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    sorted_scores = scores[order]

    # Average ranks within each group of tied scores, otherwise AUROC is biased
    # by the arbitrary order of equal predictions.
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop

    rank_sum = float(np.sum(ranks[y_true]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(scores: np.ndarray, y_true: np.ndarray) -> float:
    """AUPRC as step-wise average precision.

    Matches ``sklearn.metrics.average_precision_score``:
    ``AP = sum_n (R_n - R_{n-1}) * P_n`` over distinct score thresholds. The
    trapezoidal alternative (``auc(recall, precision)``) is optimistically
    biased on imbalanced data, which is exactly the regime this project runs in.
    """
    scores = np.asarray(scores, dtype=np.float64)
    y_true = np.asarray(y_true).astype(bool)
    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_true = y_true[order]
    sorted_scores = scores[order]

    tp = np.cumsum(sorted_true)
    fp = np.cumsum(~sorted_true)

    # Keep only the last index of each run of tied scores: a threshold cannot
    # split tied predictions.
    distinct = np.where(np.diff(sorted_scores))[0]
    keep = np.r_[distinct, sorted_scores.size - 1]

    tp = tp[keep]
    fp = fp[keep]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos

    previous_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous_recall) * precision))


def _positive_f1(precision: float, recall: float) -> float:
    """F1 for the positive class, distinguishing "failed" from "undefined".

    Two different zero-denominator situations must not be conflated:

    * **No positive samples exist** (``recall`` is nan). Nothing could be
      detected, so F1 is genuinely undefined and must be excluded from the
      macro average. Averaging it in as 0 is the legacy bug.
    * **Nothing was predicted positive**, but positives do exist (``precision``
      is nan, ``recall`` is 0). The model failed. ``F1 = 2PR/(P+R) -> 0`` as
      ``R -> 0`` for any P, so F1 is 0 and must stay in the macro average --
      excluding it would let a model that predicts nothing score a perfect
      macro F1, which is the inverse of the bug being fixed.
    """
    if np.isnan(recall):
        return float("nan")
    if recall == 0:
        return 0.0
    if np.isnan(precision):
        # recall > 0 implies tp > 0 implies tp + fp > 0, so precision is
        # defined. Unreachable; guard rather than emit a silent nan.
        raise AssertionError("precision undefined while recall > 0")
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class PathologyMetrics:
    """Every per-pathology number, for one pathology."""

    name: str
    support_positive: int
    support_negative: int
    support_valid: int
    prevalence: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    specificity: float
    npv: float
    fpr: float
    fnr: float
    auroc: float
    auprc: float
    threshold: float
    skipped_reasons: list[str] = field(default_factory=list)

    @property
    def sensitivity(self) -> float:
        """Alias for recall, under its clinical name."""
        return self.recall

    @property
    def ppv(self) -> float:
        """Alias for precision, under its clinical name."""
        return self.precision

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathology": self.name,
            "support_positive": self.support_positive,
            "support_negative": self.support_negative,
            "support_valid": self.support_valid,
            "prevalence": self.prevalence,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "ppv": self.ppv,
            "npv": self.npv,
            "fpr": self.fpr,
            "fnr": self.fnr,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "threshold": self.threshold,
            "skipped_reasons": list(self.skipped_reasons),
        }


@dataclass
class ClassificationReport:
    """Aggregate metrics plus the per-pathology breakdown."""

    per_pathology: list[PathologyMetrics]
    aggregates: dict[str, float]
    three_class_confusion: np.ndarray
    binary_confusion_matrices: dict[str, np.ndarray]
    macro_pathologies: list[str]
    skipped: dict[str, list[str]]
    settings: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "aggregates": self.aggregates,
            "per_pathology": [m.to_dict() for m in self.per_pathology],
            "macro_pathologies": self.macro_pathologies,
            "skipped": self.skipped,
        }


def _nanmean(values: list[float]) -> float:
    """Mean over defined entries. ``nan`` when nothing is defined."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan")
    return float(np.nanmean(array))


def three_class_confusion_matrices(
    labels: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    """``[P, C, C]`` confusion counts indexed ``(pathology, true, predicted)``."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    num_pathologies = labels.shape[1]
    matrices = np.zeros((num_pathologies, num_classes, num_classes), dtype=np.int64)
    for p in range(num_pathologies):
        valid = (labels[:, p] >= 0) & (labels[:, p] < num_classes)
        if not np.any(valid):
            continue
        flat = labels[valid, p] * num_classes + predictions[valid, p]
        counts = np.bincount(flat, minlength=num_classes * num_classes)
        matrices[p] = counts.reshape(num_classes, num_classes)
    return matrices


def apply_thresholds(
    positive_probabilities: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """``[N, P]`` binary predictions from per-pathology thresholds."""
    positive_probabilities = np.asarray(positive_probabilities, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (positive_probabilities.shape[1],):
        raise SchemaError(
            f"expected {positive_probabilities.shape[1]} thresholds, got "
            f"{thresholds.shape}"
        )
    return (positive_probabilities >= thresholds[None, :]).astype(np.int64)


def evaluate_classification(
    predictions: ClassificationPredictions,
    *,
    thresholds: dict[str, float] | None = None,
    uncertain_policy: str = DEFAULT_POLICY,
    include_meta_labels: bool = False,
    binary_predictions: np.ndarray | None = None,
) -> ClassificationReport:
    """Compute the full Stage-1 metric suite.

    Parameters
    ----------
    thresholds:
        Per-pathology positive-class thresholds. Pathologies absent from the
        mapping fall back to 0.5. Ignored when ``binary_predictions`` is given.
    binary_predictions:
        ``[N, P]`` pre-computed hard predictions, for reproducing the legacy
        ``argmax`` decision rule exactly.
    include_meta_labels:
        Whether ``No Finding`` and ``Support Devices`` join the macro averages.
        They are always reported per-pathology.
    """
    validate_policy(uncertain_policy)

    labels = predictions.labels
    positive_probabilities = predictions.positive_probabilities
    names = predictions.pathology_names

    threshold_vector = np.full(predictions.num_pathologies, 0.5, dtype=np.float64)
    if thresholds:
        unknown = set(thresholds) - set(names)
        if unknown:
            logger.warning(
                "threshold file names %d pathologies not in this prediction "
                "file and they are ignored: %s",
                len(unknown),
                ", ".join(sorted(unknown)),
            )
        for index, name in enumerate(names):
            if name in thresholds:
                threshold_vector[index] = float(thresholds[name])

    if binary_predictions is None:
        y_pred = apply_thresholds(positive_probabilities, threshold_vector)
    else:
        y_pred = np.asarray(binary_predictions, dtype=np.int64)
        if y_pred.shape != labels.shape:
            raise SchemaError(
                f"binary_predictions shape {y_pred.shape} != labels shape "
                f"{labels.shape}"
            )

    y_true, valid = binarize_labels(labels, uncertain_policy)

    macro_indices = predictions.macro_pathology_indices(include_meta_labels)
    per_pathology: list[PathologyMetrics] = []
    skipped: dict[str, list[str]] = {}

    micro_tp = micro_fp = micro_tn = micro_fn = 0

    for index, name in enumerate(names):
        column_valid = valid[:, index]
        yt = y_true[column_valid, index]
        yp = y_pred[column_valid, index]
        scores = positive_probabilities[column_valid, index]

        n_pos = int(np.sum(yt == 1))
        n_neg = int(np.sum(yt == 0))
        reasons: list[str] = []
        if yt.size == 0:
            reasons.append(NO_VALID_SAMPLES)
        if n_pos == 0:
            reasons.append(NO_POSITIVE_SAMPLES)
        if n_neg == 0:
            reasons.append(NO_NEGATIVE_SAMPLES)

        tp, fp, tn, fn = binary_confusion(yt, yp)
        if tp + fp == 0 and NO_VALID_SAMPLES not in reasons:
            reasons.append(NEVER_PREDICTED_POSITIVE)

        if index in macro_indices:
            micro_tp += tp
            micro_fp += fp
            micro_tn += tn
            micro_fn += fn

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _positive_f1(precision, recall)

        per_pathology.append(
            PathologyMetrics(
                name=name,
                support_positive=n_pos,
                support_negative=n_neg,
                support_valid=int(yt.size),
                prevalence=_safe_divide(n_pos, yt.size),
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
                precision=precision,
                recall=recall,
                f1=f1,
                specificity=_safe_divide(tn, tn + fp),
                npv=_safe_divide(tn, tn + fn),
                fpr=_safe_divide(fp, fp + tn),
                fnr=_safe_divide(fn, fn + tp),
                auroc=roc_auc(scores, yt) if yt.size else float("nan"),
                auprc=average_precision(scores, yt) if yt.size else float("nan"),
                threshold=float(threshold_vector[index]),
                skipped_reasons=reasons,
            )
        )
        if reasons:
            skipped[name] = reasons

    macro_metrics = [per_pathology[i] for i in macro_indices]

    micro_precision = _safe_divide(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_divide(micro_tp, micro_tp + micro_fn)
    micro_f1 = (
        float("nan")
        if np.isnan(micro_precision) or np.isnan(micro_recall)
        else _harmonic(micro_precision, micro_recall)
    )

    supports = np.asarray([m.support_positive for m in macro_metrics], dtype=np.float64)
    weight_total = float(np.sum(supports))

    def weighted(attribute: str) -> float:
        values = np.asarray(
            [getattr(m, attribute) for m in macro_metrics], dtype=np.float64
        )
        defined = ~np.isnan(values)
        if weight_total == 0 or not np.any(defined):
            return float("nan")
        w = supports[defined]
        if float(np.sum(w)) == 0:
            return float("nan")
        return float(np.sum(values[defined] * w) / np.sum(w))

    # Element-wise accuracy over the three-class problem, restricted to valid
    # (sample, pathology) pairs. Reported because it is the number a reader
    # expects to see -- and because the all-negative baseline makes its
    # uselessness on imbalanced data obvious.
    argmax_predictions = predictions.probabilities.argmax(axis=-1)
    valid_three_class = labels >= 0
    if include_meta_labels:
        column_selector = np.ones(predictions.num_pathologies, dtype=bool)
    else:
        column_selector = np.zeros(predictions.num_pathologies, dtype=bool)
        column_selector[macro_indices] = True
    selected = valid_three_class & column_selector[None, :]
    overall_accuracy = _safe_divide(
        int(np.sum((labels == argmax_predictions) & selected)), int(np.sum(selected))
    )

    binary_valid = valid & column_selector[None, :]
    binary_accuracy = _safe_divide(
        int(np.sum((y_true == y_pred) & binary_valid)), int(np.sum(binary_valid))
    )

    sensitivities = [m.recall for m in macro_metrics]
    specificities = [m.specificity for m in macro_metrics]
    balanced = [
        (s + sp) / 2.0
        for s, sp in zip(sensitivities, specificities, strict=True)
        if not (np.isnan(s) or np.isnan(sp))
    ]

    # The general macro/micro/weighted family is computed over ALL THREE
    # classes from the argmax confusion matrix, matching what the legacy
    # evaluator meant by ``f1_macro``. The positive_* family below is the
    # binary positive-vs-rest view under the configured policy and thresholds.
    # Keeping them separate is the point: in a pure binary framing the two
    # collapse onto each other and the distinction the spec asks for is lost.
    three_class = _three_class_aggregates(
        three_class_confusion_matrices(
            labels, argmax_predictions, predictions.num_classes
        ),
        macro_indices,
    )

    aggregates = {
        "accuracy": overall_accuracy,
        "binary_accuracy": binary_accuracy,
        "balanced_accuracy": _nanmean(balanced) if balanced else float("nan"),
        "macro_precision": three_class["macro_precision"],
        "macro_recall": three_class["macro_recall"],
        "macro_f1": three_class["macro_f1"],
        # In a single-label multi-class problem micro-P == micro-R == micro-F1
        # == accuracy. Reported for completeness; do not read it as independent
        # evidence.
        "micro_precision": three_class["micro"],
        "micro_recall": three_class["micro"],
        "micro_f1": three_class["micro"],
        "weighted_precision": three_class["weighted_precision"],
        "weighted_recall": three_class["weighted_recall"],
        "weighted_f1": three_class["weighted_f1"],
        "positive_macro_precision": _nanmean([m.precision for m in macro_metrics]),
        "positive_macro_recall": _nanmean([m.recall for m in macro_metrics]),
        "positive_macro_f1": _nanmean([m.f1 for m in macro_metrics]),
        "positive_micro_precision": micro_precision,
        "positive_micro_recall": micro_recall,
        "positive_micro_f1": micro_f1,
        "positive_weighted_precision": weighted("precision"),
        "positive_weighted_recall": weighted("recall"),
        "positive_weighted_f1": weighted("f1"),
        "macro_auroc": _nanmean([m.auroc for m in macro_metrics]),
        "macro_auprc": _nanmean([m.auprc for m in macro_metrics]),
        "macro_specificity": _nanmean(specificities),
        "macro_npv": _nanmean([m.npv for m in macro_metrics]),
    }

    micro_auroc, micro_auprc = _micro_probability_metrics(
        positive_probabilities, y_true, valid, macro_indices
    )
    aggregates["micro_auroc"] = micro_auroc
    aggregates["micro_auprc"] = micro_auprc

    if skipped:
        logger.info(
            "%d/%d pathologies have at least one undefined metric: %s",
            len(skipped),
            len(names),
            "; ".join(f"{k}={','.join(v)}" for k, v in sorted(skipped.items())),
        )

    binary_matrices = {
        m.name: np.array([[m.tn, m.fp], [m.fn, m.tp]], dtype=np.int64)
        for m in per_pathology
    }

    return ClassificationReport(
        per_pathology=per_pathology,
        aggregates=aggregates,
        three_class_confusion=three_class_confusion_matrices(
            labels, argmax_predictions, predictions.num_classes
        ),
        binary_confusion_matrices=binary_matrices,
        macro_pathologies=[names[i] for i in macro_indices],
        skipped=skipped,
        settings={
            "uncertain_policy": uncertain_policy,
            "uncertain_policy_description": describe_policy(uncertain_policy),
            "include_meta_labels": include_meta_labels,
            "num_samples": predictions.num_samples,
            "num_pathologies": predictions.num_pathologies,
            "class_names": list(CLASS_NAMES),
            "thresholds": {
                name: float(threshold_vector[i]) for i, name in enumerate(names)
            },
            "decision_rule": "argmax" if binary_predictions is not None else "threshold",
        },
    )


def _three_class_aggregates(
    matrices: np.ndarray, macro_indices: list[int]
) -> dict[str, float]:
    """Macro/micro/weighted P/R/F1 over all classes of the 3-class problem.

    ``matrices`` is ``[P, C, C]`` indexed ``(pathology, true, predicted)``.
    Per-class precision/recall are computed inside each pathology, averaged over
    classes, then averaged over the selected pathologies. A class with no
    support in a pathology contributes ``nan`` and is excluded rather than
    averaged in as zero.
    """
    if not macro_indices:
        nan = float("nan")
        return {
            "macro_precision": nan,
            "macro_recall": nan,
            "macro_f1": nan,
            "weighted_precision": nan,
            "weighted_recall": nan,
            "weighted_f1": nan,
            "micro": nan,
        }

    selected = matrices[macro_indices]
    per_pathology_precision: list[float] = []
    per_pathology_recall: list[float] = []
    per_pathology_f1: list[float] = []
    weighted_precision: list[float] = []
    weighted_recall: list[float] = []
    weighted_f1: list[float] = []

    for matrix in selected:
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        true_positive = np.diag(matrix)

        precisions = [_safe_divide(true_positive[c], predicted[c]) for c in range(matrix.shape[0])]
        recalls = [_safe_divide(true_positive[c], support[c]) for c in range(matrix.shape[0])]
        f1s = [
            float("nan")
            if (np.isnan(p) or np.isnan(r))
            else _harmonic(p, r)
            for p, r in zip(precisions, recalls, strict=True)
        ]

        per_pathology_precision.append(_nanmean(precisions))
        per_pathology_recall.append(_nanmean(recalls))
        per_pathology_f1.append(_nanmean(f1s))

        total = float(support.sum())
        if total > 0:
            weighted_precision.append(_weighted_by(precisions, support))
            weighted_recall.append(_weighted_by(recalls, support))
            weighted_f1.append(_weighted_by(f1s, support))

    total_correct = float(sum(np.trace(m) for m in selected))
    total_count = float(sum(m.sum() for m in selected))

    return {
        "macro_precision": _nanmean(per_pathology_precision),
        "macro_recall": _nanmean(per_pathology_recall),
        "macro_f1": _nanmean(per_pathology_f1),
        "weighted_precision": _nanmean(weighted_precision),
        "weighted_recall": _nanmean(weighted_recall),
        "weighted_f1": _nanmean(weighted_f1),
        "micro": _safe_divide(total_correct, total_count),
    }


def _weighted_by(values: list[float], weights: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    defined = ~np.isnan(array)
    if not np.any(defined) or float(np.sum(weights[defined])) == 0:
        return float("nan")
    return float(
        np.sum(array[defined] * weights[defined]) / np.sum(weights[defined])
    )


def _harmonic(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _micro_probability_metrics(
    positive_probabilities: np.ndarray,
    y_true: np.ndarray,
    valid: np.ndarray,
    macro_indices: list[int],
) -> tuple[float, float]:
    """Pool every valid (sample, pathology) pair into one ranking problem.

    Micro AUROC/AUPRC are meaningful only if the positive-class scores are
    comparable across pathologies, which they are not guaranteed to be. They are
    reported because the spec asks for them, and flagged here so a reader does
    not over-interpret them relative to the macro values.
    """
    if not macro_indices:
        return float("nan"), float("nan")
    selector = np.zeros(valid.shape[1], dtype=bool)
    selector[macro_indices] = True
    mask = valid & selector[None, :]
    if not np.any(mask):
        return float("nan"), float("nan")
    scores = positive_probabilities[mask]
    truth = y_true[mask]
    return roc_auc(scores, truth), average_precision(scores, truth)
