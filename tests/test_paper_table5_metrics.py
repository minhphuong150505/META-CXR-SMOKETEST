from __future__ import annotations

import numpy as np
import pytest

from training.evaluation.paper_table5 import (
    PAPER_TABLE5_PATHOLOGIES,
    paper_table5_weighted_f1,
)


PATHOLOGY_NAMES = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)


def test_paper_table5_metric_is_three_class_weighted_f1_over_five_pathologies():
    names = PATHOLOGY_NAMES
    labels = np.zeros((3, len(names)), dtype=np.int64)
    logits = np.zeros((3, len(names), 3), dtype=np.float64)
    for pathology in PAPER_TABLE5_PATHOLOGIES:
        labels[:, names.index(pathology)] = (0, 1, 2)
    for row in range(3):
        logits[row, :, row] = 1.0

    perfect = paper_table5_weighted_f1(labels, logits, names)
    assert perfect["mean_weighted_f1"] == 1.0

    # For one pathology, always predicting class 0 has weighted F1 = 1/6:
    # class 0 has F1=1/2 and support=1; classes 1 and 2 have F1=0.
    atelectasis = names.index("Atelectasis")
    logits[:, atelectasis, :] = 0.0
    logits[:, atelectasis, 0] = 1.0
    result = paper_table5_weighted_f1(labels, logits, names)

    assert result["per_pathology"]["Atelectasis"]["weighted_f1"] == pytest.approx(1 / 6)
    assert result["mean_weighted_f1"] == pytest.approx((4 + 1 / 6) / 5)
