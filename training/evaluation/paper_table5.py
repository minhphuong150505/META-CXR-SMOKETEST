"""Paper-compatible Table-5 metric for META-CXR encoder sensitivity."""

from __future__ import annotations

import numpy as np


# Table 5 in the META-CXR paper reports the mean F1 over these five common
# abnormalities. It is a three-class *weighted* F1, unlike positive-only macro
# F1 reported by the project's standard clinical evaluator.
PAPER_TABLE5_PATHOLOGIES = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
)


def paper_table5_weighted_f1(
    labels: np.ndarray,
    logits: np.ndarray,
    pathology_names: tuple[str, ...],
) -> dict:
    """Return Table-5's mean per-pathology weighted three-class F1.

    This is equivalent to applying ``sklearn.metrics.f1_score`` with
    ``average='weighted'`` and ``zero_division=1`` to each of the five paper
    pathologies, then taking an unweighted mean of the five scores.
    """
    indices = {name: index for index, name in enumerate(pathology_names)}
    missing = set(PAPER_TABLE5_PATHOLOGIES) - set(indices)
    if missing:
        raise ValueError(
            f"Table-5 pathologies missing from prediction schema: {sorted(missing)}"
        )

    predictions = logits.argmax(axis=-1)
    per_pathology = {}
    for name in PAPER_TABLE5_PATHOLOGIES:
        index = indices[name]
        truth = labels[:, index]
        predicted = predictions[:, index]
        valid = (truth >= 0) & (truth < 3)
        truth = truth[valid]
        predicted = predicted[valid]
        if truth.size == 0:
            raise ValueError(f"No valid Table-5 labels for {name}")

        weighted_sum = 0.0
        class_support = {}
        for class_index in range(3):
            true_class = truth == class_index
            predicted_class = predicted == class_index
            support = int(true_class.sum())
            tp = int((true_class & predicted_class).sum())
            fp = int((~true_class & predicted_class).sum())
            fn = int((true_class & ~predicted_class).sum())
            precision = 1.0 if tp + fp == 0 else tp / (tp + fp)
            recall = 1.0 if tp + fn == 0 else tp / (tp + fn)
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            weighted_sum += support * f1
            class_support[str(class_index)] = support

        per_pathology[name] = {
            "weighted_f1": weighted_sum / truth.size,
            "valid_samples": int(truth.size),
            "class_support": class_support,
        }

    return {
        "metric": "mean_per_pathology_weighted_f1_three_class_argmax",
        "pathologies": list(PAPER_TABLE5_PATHOLOGIES),
        "mean_weighted_f1": float(np.mean([
            per_pathology[name]["weighted_f1"] for name in PAPER_TABLE5_PATHOLOGIES
        ])),
        "per_pathology": per_pathology,
    }
