import torch
import torch.nn.functional as F

from mhcac.loss import ClassificationLoss, soft_target_kl_loss
from mhcac.mhcac_12 import AbnormalityClassificationModel
from vision_encoders.shared_visual_tokens import SharedVisualTokens


def test_classification_loss_masks_unlabelled_samples():
    torch.manual_seed(0)
    logits = torch.randn(3, 2, 3, requires_grad=True)
    labels = torch.tensor([[0, 1], [2, 0], [1, 2]])
    mask = torch.tensor([True, False, True])

    loss_fn = ClassificationLoss(num_abnormalities=2, label_smoothing=0.0)
    actual = loss_fn(logits, labels, sample_mask=mask)
    expected = torch.stack(
        [F.cross_entropy(logits[mask, i], labels[mask, i]) for i in range(2)]
    ).mean()

    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.count_nonzero(logits.grad[1]) == 0


def test_empty_classification_mask_returns_differentiable_zero():
    logits = torch.randn(2, 2, 3, requires_grad=True)
    labels = torch.zeros(2, 2, dtype=torch.long)
    loss_fn = ClassificationLoss(num_abnormalities=2)

    loss = loss_fn(logits, labels, sample_mask=torch.zeros(2, dtype=torch.bool))

    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_distillation_detaches_teacher_and_masks_invalid_rows():
    student = torch.randn(3, 2, 3, requires_grad=True)
    teacher = torch.randn(3, 2, 3, requires_grad=True)
    mask = torch.tensor([True, False, True])

    loss = soft_target_kl_loss(student, teacher, mask, temperature=2.0)
    loss.backward()

    assert student.grad is not None
    assert torch.count_nonzero(student.grad[1]) == 0
    assert teacher.grad is None


def test_mhcac_text_is_teacher_only_and_student_shape_matches_inference():
    torch.manual_seed(1)
    model = AbnormalityClassificationModel(
        embed_dim=16,
        num_heads=4,
        num_abnormalities=3,
        num_classes=3,
        num_layers=2,
        num_commmon_tokens=3,
        visual_dim=16,
        txt_dim=16,
        target_patch_count=4,
        text_dropout_rate=0.2,
        use_cnn=False,
    ).eval()
    shared = SharedVisualTokens(
        tokens=torch.randn(2, 4, 16), spans={"pubmedclip": slice(0, 4)}
    )
    text_embeddings = torch.randn(2, 5, 16)
    labels = torch.tensor([[0, 1, 2], [1, 0, 0]])

    text_attention_calls = []
    handle = model.attention_layers[0].expert_to_text_attention.register_forward_hook(
        lambda *args: text_attention_calls.append(True)
    )
    student_logits, _, contrastive, orthogonal, sparse = model(
        shared,
        text_embeddings=None,
        labels=labels,
    )
    assert text_attention_calls == []
    teacher_logits, _, _, _, _ = model(
        shared,
        text_embeddings=text_embeddings,
        labels=labels,
    )
    handle.remove()

    assert text_attention_calls
    assert student_logits.shape == teacher_logits.shape == (2, 3, 3)
    for value in (student_logits, teacher_logits, contrastive, orthogonal, sparse):
        assert torch.isfinite(value).all()
