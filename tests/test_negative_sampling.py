import pytest
import torch

from smoke.sampling import negative_sampling_weights


def test_masks_dominant_positive_before_softmax():
    similarity = torch.tensor([[1000.0, 0.0, -1.0]])
    candidates = torch.tensor([[False, True, True]])

    # The old softmax-then-mask implementation produced [0, 0, 0] here because
    # exp(-1000) underflows. The negative-only softmax must remain sampleable.
    weights = negative_sampling_weights(similarity, candidates)

    assert weights.sum().item() == pytest.approx(1.0)
    assert weights[0, 0].item() == 0.0
    assert weights[0, 1].item() > weights[0, 2].item() > 0.0
    assert torch.multinomial(weights[0], 1).item() in {1, 2}


def test_nonfinite_scores_have_safe_fallbacks():
    similarity = torch.tensor(
        [[float("nan"), float("-inf"), float("nan")],
         [float("inf"), float("inf"), 3.0]]
    )
    candidates = torch.tensor([[False, True, True], [True, True, True]])

    weights = negative_sampling_weights(similarity, candidates)

    assert torch.isfinite(weights).all()
    assert weights.sum(dim=1).tolist() == pytest.approx([1.0, 1.0])
    assert weights[0].tolist() == pytest.approx([0.0, 0.5, 0.5])
    assert weights[1].tolist() == pytest.approx([0.5, 0.5, 0.0])


def test_empty_candidate_row_fails_before_cuda_multinomial():
    with pytest.raises(RuntimeError, match="no eligible candidate.*rows.*0"):
        negative_sampling_weights(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
