"""Trivial baselines, for detecting deceptively high metrics.

On MIMIC-CXR most (study, pathology) pairs are negative, so a model that never
predicts a positive finding still scores high accuracy. A metric table without
baselines cannot distinguish "the model learned something" from "the split is
imbalanced". Every baseline here is computed from the labels alone -- none of
them looks at the model's probabilities except ``threshold_half``, which reuses
them at the fixed default threshold.

Reading the table
-----------------
If the model's accuracy is close to ``all_negative`` while its positive macro F1
is close to 0, the model has not learned to detect findings, whatever the
headline accuracy says.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from training.evaluation.classification_metrics import (
    ClassificationReport,
    evaluate_classification,
)
from training.evaluation.schemas import ClassificationPredictions
from training.evaluation.uncertain_policy import DEFAULT_POLICY, binarize_labels

logger = logging.getLogger(__name__)

ALL_NEGATIVE = "all_negative"
ALL_POSITIVE = "all_positive"
MAJORITY_CLASS = "majority_class"
PREVALENCE_RANDOM = "prevalence_random"
THRESHOLD_HALF = "threshold_half"

BASELINES = (ALL_NEGATIVE, ALL_POSITIVE, MAJORITY_CLASS, PREVALENCE_RANDOM, THRESHOLD_HALF)


@dataclass
class BaselineRow:
    """One row of the baseline comparison table."""

    name: str
    accuracy: float
    positive_macro_f1: float
    positive_macro_recall: float
    macro_auroc: float
    macro_auprc: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.name,
            "accuracy": self.accuracy,
            "positive_macro_f1": self.positive_macro_f1,
            "positive_macro_recall": self.positive_macro_recall,
            "macro_auroc": self.macro_auroc,
            "macro_auprc": self.macro_auprc,
            "description": self.description,
        }


def _row(name: str, report: ClassificationReport, description: str) -> BaselineRow:
    aggregates = report.aggregates
    return BaselineRow(
        name=name,
        accuracy=aggregates["binary_accuracy"],
        positive_macro_f1=aggregates["positive_macro_f1"],
        positive_macro_recall=aggregates["positive_macro_recall"],
        macro_auroc=aggregates["macro_auroc"],
        macro_auprc=aggregates["macro_auprc"],
        description=description,
    )


def compute_baselines(
    predictions: ClassificationPredictions,
    *,
    uncertain_policy: str = DEFAULT_POLICY,
    include_meta_labels: bool = False,
    seed: int = 42,
    which: tuple[str, ...] = BASELINES,
) -> list[BaselineRow]:
    """Evaluate each trivial baseline under the same settings as the model.

    AUROC/AUPRC are reported for completeness but are meaningless for the
    constant baselines (their scores carry no ranking information); they appear
    as ``nan`` or 0.5-like values and should not be compared.
    """
    y_true, valid = binarize_labels(predictions.labels, uncertain_policy)
    shape = predictions.labels.shape
    rows: list[BaselineRow] = []

    def evaluate(binary: np.ndarray) -> ClassificationReport:
        return evaluate_classification(
            predictions,
            uncertain_policy=uncertain_policy,
            include_meta_labels=include_meta_labels,
            binary_predictions=binary,
        )

    if ALL_NEGATIVE in which:
        rows.append(
            _row(
                ALL_NEGATIVE,
                evaluate(np.zeros(shape, dtype=np.int64)),
                "predict negative for every pathology on every study",
            )
        )

    if ALL_POSITIVE in which:
        rows.append(
            _row(
                ALL_POSITIVE,
                evaluate(np.ones(shape, dtype=np.int64)),
                "predict positive for every pathology on every study",
            )
        )

    if MAJORITY_CLASS in which:
        majority = np.zeros(shape, dtype=np.int64)
        for index in range(shape[1]):
            column_valid = valid[:, index]
            if not np.any(column_valid):
                continue
            positives = int(np.sum(y_true[column_valid, index] == 1))
            if positives * 2 > int(np.sum(column_valid)):
                majority[:, index] = 1
        rows.append(
            _row(
                MAJORITY_CLASS,
                evaluate(majority),
                "predict each pathology's majority class, decided per pathology",
            )
        )

    if PREVALENCE_RANDOM in which:
        rng = np.random.default_rng(seed)
        random_predictions = np.zeros(shape, dtype=np.int64)
        for index in range(shape[1]):
            column_valid = valid[:, index]
            n_valid = int(np.sum(column_valid))
            prevalence = (
                float(np.sum(y_true[column_valid, index] == 1)) / n_valid
                if n_valid
                else 0.0
            )
            random_predictions[:, index] = (
                rng.random(shape[0]) < prevalence
            ).astype(np.int64)
        rows.append(
            _row(
                PREVALENCE_RANDOM,
                evaluate(random_predictions),
                f"sample each prediction from the pathology's prevalence (seed={seed})",
            )
        )

    if THRESHOLD_HALF in which:
        rows.append(
            _row(
                THRESHOLD_HALF,
                evaluate_classification(
                    predictions,
                    uncertain_policy=uncertain_policy,
                    include_meta_labels=include_meta_labels,
                    thresholds=None,
                ),
                "the model's own probabilities at the uncalibrated 0.5 threshold",
            )
        )

    return rows


def baseline_table(
    rows: list[BaselineRow], model_row: BaselineRow | None = None
) -> str:
    """Render the comparison as a Markdown table."""
    header = (
        "| Model | Accuracy | Positive Macro F1 | Macro AUROC | Macro AUPRC |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
    )

    def fmt(value: float) -> str:
        return "n/a" if np.isnan(value) else f"{value:.4f}"

    body = ""
    if model_row is not None:
        body += (
            f"| **{model_row.name}** | {fmt(model_row.accuracy)} | "
            f"{fmt(model_row.positive_macro_f1)} | {fmt(model_row.macro_auroc)} | "
            f"{fmt(model_row.macro_auprc)} |\n"
        )
    for row in rows:
        body += (
            f"| {row.name} | {fmt(row.accuracy)} | {fmt(row.positive_macro_f1)} | "
            f"{fmt(row.macro_auroc)} | {fmt(row.macro_auprc)} |\n"
        )
    return header + body
