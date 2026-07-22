"""Stage-1 classification metric tests.

Every expected value here is computed by hand in the test body or in a comment,
so a regression shows up as a wrong number rather than as "the code still runs".
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.classification_metrics import (  # noqa: E402
    NEVER_PREDICTED_POSITIVE,
    NO_NEGATIVE_SAMPLES,
    NO_POSITIVE_SAMPLES,
    average_precision,
    binary_confusion,
    evaluate_classification,
    roc_auc,
)
from training.evaluation.schemas import (  # noqa: E402
    ClassificationPredictions,
    SchemaError,
)
from training.evaluation.uncertain_policy import (  # noqa: E402
    IGNORE_UNCERTAIN,
    THREE_CLASS,
    UNCERTAIN_AS_NEGATIVE,
    UNCERTAIN_AS_POSITIVE,
    binarize_labels,
)


def make_predictions(labels, positive_probs, names=None):
    """Build a ClassificationPredictions from labels and positive scores.

    The remaining probability mass is split between negative and uncertain so
    that rows sum to 1 and ``argmax`` is well defined.
    """
    labels = np.asarray(labels)
    positive = np.asarray(positive_probs, dtype=np.float64)
    remainder = 1.0 - positive
    probabilities = np.stack([remainder * 0.7, positive, remainder * 0.3], axis=-1)
    names = names or tuple(f"P{i}" for i in range(labels.shape[1]))
    return ClassificationPredictions(
        labels=labels,
        probabilities=probabilities,
        pathology_names=names,
        sample_keys=np.asarray([f"s{i}" for i in range(labels.shape[0])]),
    )


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def test_binary_confusion_counts():
    tp, fp, tn, fn = binary_confusion([1, 0, 0, 1], [1, 1, 0, 0])
    assert (tp, fp, tn, fn) == (1, 1, 1, 1)


def test_roc_auc_perfect_and_inverted():
    assert roc_auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert roc_auc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0


def test_roc_auc_handles_ties_with_average_ranks():
    # All scores equal -> every ordering is arbitrary -> AUROC must be 0.5.
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == 0.5


def test_roc_auc_single_class_is_nan_not_half():
    # Returning 0.5 here would look like a measured coin flip.
    assert math.isnan(roc_auc([0.9, 0.8], [1, 1]))
    assert math.isnan(roc_auc([0.9, 0.8], [0, 0]))


def test_roc_auc_known_value():
    # scores 0.1(T) 0.9(F) 0.8(F) 0.7(F): the only positive ranks last.
    assert roc_auc([0.1, 0.9, 0.8, 0.7], [1, 0, 0, 0]) == 0.0


def test_average_precision_known_value():
    # One positive, ranked last of four. Precision at that point is 1/4,
    # recall goes 0 -> 1, so AP = 1 * 0.25.
    assert average_precision([0.1, 0.9, 0.8, 0.7], [1, 0, 0, 0]) == pytest.approx(0.25)


def test_average_precision_perfect_ranking():
    assert average_precision([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_average_precision_no_positive_is_nan():
    assert math.isnan(average_precision([0.9, 0.1], [0, 0]))


def test_average_precision_reflects_imbalance():
    # 1 positive in 100. A perfect ranker scores 1.0; a random-looking ranker
    # that puts the positive in the middle scores far lower, whereas AUROC
    # would still look respectable. This is why AUPRC is reported.
    scores = np.linspace(0, 1, 100)
    y = np.zeros(100, dtype=int)
    y[-1] = 1
    assert average_precision(scores, y) == pytest.approx(1.0)

    y_mid = np.zeros(100, dtype=int)
    y_mid[50] = 1
    ap = average_precision(scores, y_mid)
    auc = roc_auc(scores, y_mid)
    assert ap < 0.05
    assert auc > 0.5


# --------------------------------------------------------------------------
# The bug this module was written to fix
# --------------------------------------------------------------------------


def test_all_negative_predictions_give_high_accuracy_and_zero_positive_f1():
    """The headline failure mode: accuracy looks great, the model is useless.

    19 of 20 (sample, pathology) pairs are negative. Predicting everything
    negative gives 95% accuracy and a positive macro F1 of exactly 0.
    """
    labels = np.zeros((10, 2), dtype=int)
    labels[0, 0] = 1  # one positive in each pathology
    labels[0, 1] = 1
    positive = np.full((10, 2), 0.01)  # never crosses threshold 0.5

    report = evaluate_classification(make_predictions(labels, positive))

    assert report.aggregates["binary_accuracy"] == pytest.approx(0.9)
    assert report.aggregates["positive_macro_f1"] == 0.0
    assert report.aggregates["positive_macro_recall"] == 0.0
    # Precision is undefined (nothing was predicted positive), not zero.
    assert math.isnan(report.aggregates["positive_macro_precision"])
    for metrics in report.per_pathology:
        assert NEVER_PREDICTED_POSITIVE in metrics.skipped_reasons


def test_pathology_without_positive_samples_is_excluded_from_macro():
    """Regression guard for the legacy `f1[:, 1].mean()` dilution.

    P0 is predicted perfectly. P1 has no positive samples at all, so its
    positive F1 is undefined. The macro must be 1.0 (P0 only), not 0.5.
    """
    labels = np.array([[1, 0], [1, 0], [0, 0], [0, 0]])
    positive = np.array([[0.9, 0.1], [0.8, 0.1], [0.2, 0.1], [0.1, 0.1]])

    report = evaluate_classification(make_predictions(labels, positive))

    assert report.aggregates["positive_macro_f1"] == pytest.approx(1.0)
    assert NO_POSITIVE_SAMPLES in report.skipped["P1"]
    assert math.isnan(report.per_pathology[1].f1)
    assert math.isnan(report.per_pathology[1].auroc)


def test_pathology_without_negative_samples_is_flagged():
    labels = np.array([[1, 1], [1, 1]])
    positive = np.array([[0.9, 0.9], [0.8, 0.8]])
    report = evaluate_classification(make_predictions(labels, positive))
    assert NO_NEGATIVE_SAMPLES in report.skipped["P0"]
    # Recall is still defined; AUROC is not.
    assert report.per_pathology[0].recall == pytest.approx(1.0)
    assert math.isnan(report.per_pathology[0].auroc)


def test_evaluator_does_not_crash_when_a_class_is_absent():
    labels = np.zeros((5, 3), dtype=int)
    positive = np.full((5, 3), 0.2)
    report = evaluate_classification(make_predictions(labels, positive))
    assert len(report.per_pathology) == 3
    assert math.isnan(report.aggregates["macro_auroc"])


# --------------------------------------------------------------------------
# Macro vs micro must not be confused
# --------------------------------------------------------------------------


def test_positive_macro_differs_from_positive_micro_under_imbalance():
    """A rare pathology gets equal weight in macro, tiny weight in micro."""
    # P0: 8 samples, all classified correctly.
    # P1: 2 samples, both positives missed.
    labels = np.array([[1, 1]] * 2 + [[0, 0]] * 6)
    positive = np.zeros((8, 2))
    positive[:2, 0] = 0.9  # P0 positives found
    positive[:2, 1] = 0.1  # P1 positives missed

    report = evaluate_classification(make_predictions(labels, positive))
    macro = report.aggregates["positive_macro_f1"]
    micro = report.aggregates["positive_micro_f1"]

    # P0 F1 = 1.0, P1 F1 = 0.0 -> macro 0.5.
    assert macro == pytest.approx(0.5)
    # Micro pools counts: tp=2, fp=0, fn=2 -> P=1, R=0.5, F1=2/3.
    assert micro == pytest.approx(2 / 3)
    assert macro != micro


def test_three_class_macro_is_not_the_positive_macro():
    """The general macro family averages over all three classes."""
    labels = np.array([[1], [0], [2], [0]])
    positive = np.array([[0.9], [0.1], [0.4], [0.1]])
    report = evaluate_classification(make_predictions(labels, positive))
    assert report.aggregates["macro_f1"] != report.aggregates["positive_macro_f1"]


# --------------------------------------------------------------------------
# Uncertain-label policies
# --------------------------------------------------------------------------


def test_binarize_labels_under_each_policy():
    labels = np.array([[0, 1, 2, -1]])

    binary, valid = binarize_labels(labels, THREE_CLASS)
    assert binary.tolist() == [[0, 1, 0, 0]]
    assert valid.tolist() == [[True, True, True, False]]

    binary, valid = binarize_labels(labels, UNCERTAIN_AS_POSITIVE)
    assert binary.tolist() == [[0, 1, 1, 0]]
    assert valid.tolist() == [[True, True, True, False]]

    binary, valid = binarize_labels(labels, UNCERTAIN_AS_NEGATIVE)
    assert binary.tolist() == [[0, 1, 0, 0]]
    assert valid.tolist() == [[True, True, True, False]]

    binary, valid = binarize_labels(labels, IGNORE_UNCERTAIN)
    assert binary.tolist() == [[0, 1, 0, 0]]
    assert valid.tolist() == [[True, True, False, False]]


def test_uncertain_policy_changes_metrics_as_expected():
    """One positive, one uncertain, two negatives; all scored above threshold.

    U-Ones makes the uncertain sample a true positive; U-Zeros makes it a false
    positive; U-Ignore drops it entirely.
    """
    labels = np.array([[1], [2], [0], [0]])
    positive = np.array([[0.9], [0.9], [0.1], [0.1]])
    preds = make_predictions(labels, positive)

    as_pos = evaluate_classification(preds, uncertain_policy=UNCERTAIN_AS_POSITIVE)
    assert as_pos.per_pathology[0].tp == 2
    assert as_pos.per_pathology[0].fp == 0
    assert as_pos.aggregates["positive_macro_f1"] == pytest.approx(1.0)

    as_neg = evaluate_classification(preds, uncertain_policy=UNCERTAIN_AS_NEGATIVE)
    assert as_neg.per_pathology[0].tp == 1
    assert as_neg.per_pathology[0].fp == 1  # the uncertain sample

    ignored = evaluate_classification(preds, uncertain_policy=IGNORE_UNCERTAIN)
    assert ignored.per_pathology[0].tp == 1
    assert ignored.per_pathology[0].fp == 0
    assert ignored.per_pathology[0].support_valid == 3  # one sample dropped
    assert ignored.aggregates["positive_macro_f1"] == pytest.approx(1.0)


def test_policy_is_recorded_in_settings():
    labels = np.array([[1], [0]])
    report = evaluate_classification(
        make_predictions(labels, np.array([[0.9], [0.1]])),
        uncertain_policy=IGNORE_UNCERTAIN,
    )
    assert report.settings["uncertain_policy"] == IGNORE_UNCERTAIN
    assert "U-Ignore" in report.settings["uncertain_policy_description"]


# --------------------------------------------------------------------------
# Missing labels, thresholds, meta-labels, shape validation
# --------------------------------------------------------------------------


def test_missing_labels_are_excluded_everywhere():
    labels = np.array([[1], [-1], [0]])
    positive = np.array([[0.9], [0.9], [0.1]])
    report = evaluate_classification(make_predictions(labels, positive))
    # The -1 row would have been a false positive had it been counted.
    assert report.per_pathology[0].support_valid == 2
    assert report.per_pathology[0].fp == 0
    assert report.aggregates["positive_macro_f1"] == pytest.approx(1.0)


def test_per_pathology_thresholds_are_applied_independently():
    labels = np.array([[1, 1], [0, 0]])
    positive = np.array([[0.6, 0.6], [0.4, 0.4]])
    preds = make_predictions(labels, positive)

    # P0 threshold 0.5 separates the two; P1 threshold 0.7 predicts nothing.
    report = evaluate_classification(preds, thresholds={"P0": 0.5, "P1": 0.7})
    assert report.per_pathology[0].tp == 1
    assert report.per_pathology[1].tp == 0
    assert report.settings["thresholds"] == {"P0": 0.5, "P1": 0.7}


def test_unlisted_pathology_falls_back_to_half():
    labels = np.array([[1], [0]])
    report = evaluate_classification(
        make_predictions(labels, np.array([[0.9], [0.1]])), thresholds={}
    )
    assert report.settings["thresholds"]["P0"] == 0.5


def test_meta_labels_are_excluded_from_macro_by_default():
    names = ("No Finding", "Cardiomegaly", "Support Devices")
    labels = np.array([[1, 1, 1], [0, 0, 0]])
    positive = np.array([[0.9, 0.9, 0.9], [0.1, 0.1, 0.1]])
    preds = make_predictions(labels, positive, names=names)

    default = evaluate_classification(preds)
    assert default.macro_pathologies == ["Cardiomegaly"]
    # All three are still reported individually.
    assert len(default.per_pathology) == 3

    everything = evaluate_classification(preds, include_meta_labels=True)
    assert len(everything.macro_pathologies) == 3


def test_shape_mismatch_raises_schema_error():
    with pytest.raises(SchemaError, match="must match labels"):
        ClassificationPredictions(
            labels=np.zeros((4, 2), dtype=int),
            probabilities=np.zeros((4, 3, 3)),
            pathology_names=("a", "b"),
            sample_keys=np.asarray(["a", "b", "c", "d"]),
        )


def test_wrong_number_of_names_raises():
    with pytest.raises(SchemaError, match="pathology names"):
        ClassificationPredictions(
            labels=np.zeros((2, 2), dtype=int),
            probabilities=np.zeros((2, 2, 3)),
            pathology_names=("only_one",),
            sample_keys=np.asarray(["a", "b"]),
        )


def test_out_of_range_label_raises():
    with pytest.raises(SchemaError, match="out-of-range"):
        ClassificationPredictions(
            labels=np.array([[7]]),
            probabilities=np.zeros((1, 1, 3)),
            pathology_names=("a",),
            sample_keys=np.asarray(["a"]),
        )


def test_predictions_round_trip_through_npz(tmp_path):
    labels = np.array([[1, 0], [0, 2]])
    preds = make_predictions(labels, np.array([[0.9, 0.1], [0.2, 0.6]]))
    preds.metadata = {"split": "validation", "checkpoint": "best.pth"}

    path = preds.save(tmp_path / "val.npz")
    loaded = ClassificationPredictions.load(path)

    assert np.array_equal(loaded.labels, preds.labels)
    assert np.allclose(loaded.probabilities, preds.probabilities)
    assert loaded.pathology_names == preds.pathology_names
    assert loaded.metadata["split"] == "validation"

    # And the metrics computed from the reloaded file are identical, which is
    # the property that makes offline re-evaluation trustworthy.
    assert (
        evaluate_classification(loaded).aggregates["positive_macro_f1"]
        == evaluate_classification(preds).aggregates["positive_macro_f1"]
    )
