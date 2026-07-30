"""Guards on the Stage-1 evaluation hook in `model/lavis/tasks/image_text_pretrain.py`.

The contract being protected: adding the corrected macro metrics and the
prediction dump must **not** change any value the historical evaluator already
reported. ``f1_positive_macro`` selects checkpoints, so a changed value would
silently change which checkpoint a resumed run keeps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

image_text_pretrain = pytest.importorskip(
    "model.lavis.tasks.image_text_pretrain",
    reason="LAVIS task module needs the full model stack",
)
ImageTextPretrainTask = image_text_pretrain.ImageTextPretrainTask


class FakeModel(torch.nn.Module):
    """Returns pre-baked logits, so the metric maths is the only variable."""

    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs
        self.calls = 0
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        output = self.outputs[self.calls]
        self.calls += 1
        return output


def make_batch(labels, mask=None):
    labels = torch.tensor(labels, dtype=torch.long)
    batch = {"classification_labels": labels, "text_output": [""] * labels.shape[0]}
    if mask is not None:
        batch["classification_mask"] = torch.tensor(mask, dtype=torch.bool)
    return batch


def logits_for(predictions, num_classes=3, confidence=5.0):
    """One-hot-ish logits that argmax to `predictions`."""
    predictions = torch.tensor(predictions, dtype=torch.long)
    logits = torch.zeros(*predictions.shape, num_classes)
    return logits.scatter_(-1, predictions.unsqueeze(-1), confidence)


def run_evaluation(labels, predictions, mask=None):
    batch = make_batch(labels, mask)
    output = {"loss": torch.tensor(1.0), "classification_logits": logits_for(predictions)}
    model = FakeModel([output])
    task = ImageTextPretrainTask()
    return task.evaluation(model, [batch], cuda_enabled=False)


def test_legacy_metric_keys_are_still_reported():
    stats = run_evaluation(
        labels=[[1, 0], [0, 1], [1, 0], [0, 0]],
        predictions=[[1, 0], [0, 1], [1, 0], [0, 0]],
    )
    for key in (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "precision_positive_macro",
        "recall_positive_macro",
        "f1_positive_macro",
    ):
        assert key in stats, f"legacy key {key} disappeared"


def test_legacy_positive_macro_still_averages_over_all_pathologies():
    """The historical (diluted) value is preserved exactly.

    P0 is predicted perfectly and has positives. P1 has no positive samples, so
    its positive-class F1 is 0 under the legacy formula. The legacy macro is
    therefore (1 + 0) / 2 = 0.5 -- and must stay 0.5.
    """
    stats = run_evaluation(
        labels=[[1, 0], [1, 0], [0, 0], [0, 0]],
        predictions=[[1, 0], [1, 0], [0, 0], [0, 0]],
    )
    assert stats["f1_positive_macro"] == pytest.approx(0.5)


def test_defined_only_metric_excludes_zero_support_pathologies():
    """The corrected value for the same input is 1.0, reported separately."""
    stats = run_evaluation(
        labels=[[1, 0], [1, 0], [0, 0], [0, 0]],
        predictions=[[1, 0], [1, 0], [0, 0], [0, 0]],
    )
    assert stats["f1_positive_macro_defined_only"] == pytest.approx(1.0)
    assert stats["num_pathologies_with_positives"] == 1.0
    # The two must genuinely disagree here, otherwise the guard proves nothing.
    assert stats["f1_positive_macro"] != stats["f1_positive_macro_defined_only"]


def test_the_two_macros_agree_when_every_pathology_has_positives():
    stats = run_evaluation(
        labels=[[1, 1], [0, 0]],
        predictions=[[1, 1], [0, 0]],
    )
    assert stats["f1_positive_macro"] == pytest.approx(
        stats["f1_positive_macro_defined_only"]
    )


def test_defined_only_keys_are_absent_when_no_pathology_has_positives():
    stats = run_evaluation(labels=[[0, 0], [0, 0]], predictions=[[0, 0], [0, 0]])
    assert "f1_positive_macro" in stats
    assert "f1_positive_macro_defined_only" not in stats


def test_classification_mask_excludes_unlabelled_rows():
    """A masked row must not become a true negative."""
    masked = run_evaluation(
        labels=[[1], [0]],
        predictions=[[1], [1]],
        mask=[True, False],
    )
    unmasked = run_evaluation(
        labels=[[1], [0]],
        predictions=[[1], [1]],
        mask=[True, True],
    )
    # With the second row masked out there is no false positive left.
    assert masked["precision_positive_macro"] == pytest.approx(1.0)
    assert unmasked["precision_positive_macro"] == pytest.approx(0.5)


def test_ignore_uncertain_excludes_uncertain_label_cells():
    task = ImageTextPretrainTask()
    task.run_cfg = {"uncertain_policy": "ignore_uncertain"}
    labels = [[2], [1]]
    predictions = [[1], [1]]
    batch = make_batch(labels)
    output = {
        "loss": torch.tensor(1.0),
        "classification_logits": logits_for(predictions),
    }
    stats = task.evaluation(FakeModel([output]), [batch], cuda_enabled=False)
    assert stats["precision_positive_macro"] == pytest.approx(1.0)


def test_shape_mismatch_still_raises():
    batch = make_batch([[1, 0], [0, 1]])
    output = {
        "loss": torch.tensor(1.0),
        # 3 abnormalities where the labels declare 2
        "classification_logits": torch.zeros(2, 3, 3),
    }
    task = ImageTextPretrainTask()
    with pytest.raises(ValueError, match=r"\[B, abnormalities, classes\]"):
        task.evaluation(FakeModel([output]), [batch], cuda_enabled=False)


def test_predictions_are_not_saved_unless_requested(tmp_path):
    """Default behaviour is unchanged: no file is written."""
    task = ImageTextPretrainTask()
    assert not getattr(task, "cfg", None)
    stats = run_evaluation(labels=[[1], [0]], predictions=[[1], [0]])
    assert "f1_positive_macro" in stats
    assert list(tmp_path.iterdir()) == []


class FakeStudentModel(torch.nn.Module):
    """The image-only path; evaluation recomputes its classification loss."""

    def __init__(self, logits, loss_value=0.25):
        super().__init__()
        self.logits = logits
        self.loss_value = loss_value
        self.seen_masks = []
        self._parameter = torch.nn.Parameter(torch.zeros(1))

    def forward_image(self, batch):
        return self.logits, None

    def cls_loss_fn(self, logits, labels, sample_mask=None):
        self.seen_masks.append(sample_mask)
        return torch.tensor(self.loss_value)


def run_student_evaluation(labels, predictions, mask=None, loss_value=0.25):
    batch = make_batch(labels, mask)
    model = FakeStudentModel(logits_for(predictions), loss_value=loss_value)
    task = ImageTextPretrainTask()
    task.run_cfg = {"classification_only_eval": True}
    return task.evaluation(model, [batch], cuda_enabled=False), model


def test_classification_only_eval_reports_a_selection_loss():
    """Without this, `selection_metric: loss_cls` has nothing to select on.

    forward_image returns logits only, so before the loss was recomputed here the
    validation stats carried no loss key at all.
    """
    stats, _ = run_student_evaluation(
        labels=[[1, 0], [0, 1]], predictions=[[1, 0], [0, 1]], loss_value=0.5
    )
    assert stats["loss_cls"] == pytest.approx(0.5)


def test_selection_loss_uses_the_same_mask_as_training():
    """Rows without a CheXpert label are masked out of the loss, not counted."""
    _, model = run_student_evaluation(
        labels=[[1, 0], [0, 1]],
        predictions=[[1, 0], [0, 1]],
        mask=[True, False],
    )
    assert model.seen_masks[0].tolist() == [True, False]


def test_full_forward_eval_still_reports_its_own_losses():
    """classification_only_eval: false keeps the multi-objective loss keys."""
    stats = run_evaluation(labels=[[1, 0], [0, 1]], predictions=[[1, 0], [0, 1]])
    assert stats["loss"] == pytest.approx(1.0)
    assert "loss_cls" not in stats
