"""Unit tests for the multi-view auxiliary losses.

Run with `python -m tests.test_multiview_losses`. No GPU required.
"""
import torch

from mhcac.loss import MultiPositiveContrastiveLoss, view_consistency_loss


B, N, P, D = 4, 2, 5, 16


def test_mpc_zero_without_aux():
    """No auxiliary view anywhere -> no positives -> exactly zero, no NaN."""
    loss_fn = MultiPositiveContrastiveLoss()
    anchor = torch.randn(B, P, D)
    aux = torch.randn(B, N, P, D)
    out = loss_fn(anchor, aux, torch.zeros(B, N, dtype=torch.bool))
    assert torch.isfinite(out) and out.item() == 0.0, out
    assert loss_fn(anchor, torch.zeros(B, 0, P, D), torch.zeros(B, 0, dtype=torch.bool)).item() == 0.0
    print("ok: MPC is exactly zero when no study has an auxiliary view")


def test_mpc_rewards_same_study_similarity():
    """Aux views that match their own anchor score a lower loss than shuffled ones."""
    loss_fn = MultiPositiveContrastiveLoss()
    torch.manual_seed(0)
    anchor = torch.randn(B, P, D)
    aux_mask = torch.ones(B, N, dtype=torch.bool)

    aligned = anchor.unsqueeze(1).repeat(1, N, 1, 1) + 0.01 * torch.randn(B, N, P, D)
    shuffled = aligned[torch.tensor([1, 2, 3, 0])]

    l_aligned = loss_fn(anchor, aligned, aux_mask)
    l_shuffled = loss_fn(anchor, shuffled, aux_mask)
    assert torch.isfinite(l_aligned) and torch.isfinite(l_shuffled)
    assert l_aligned < l_shuffled, (l_aligned.item(), l_shuffled.item())
    print(
        f"ok: MPC lower for same-study positives "
        f"({l_aligned.item():.4f} < {l_shuffled.item():.4f})"
    )


def test_mpc_partial_mask_is_finite():
    """Ragged per-study auxiliary counts stay finite and differentiable."""
    loss_fn = MultiPositiveContrastiveLoss()
    anchor = torch.randn(B, P, D, requires_grad=True)
    aux = torch.randn(B, N, P, D)
    aux_mask = torch.tensor([[True, True], [True, False], [False, False], [True, False]])
    out = loss_fn(anchor, aux, aux_mask)
    assert torch.isfinite(out), out
    out.backward()
    assert anchor.grad is not None and torch.isfinite(anchor.grad).all()
    print("ok: MPC handles ragged masks, stays finite, backprops")


def test_view_consistency_zero_when_identical():
    """Identical logits -> zero divergence; different logits -> positive."""
    logits = torch.randn(B, 14, 3)
    has_aux = torch.tensor([True, True, False, False])
    same = view_consistency_loss(logits, logits.clone(), has_aux)
    assert abs(same.item()) < 1e-6, same

    other = view_consistency_loss(logits, torch.randn(B, 14, 3), has_aux)
    assert other.item() > 0, other

    none_ = view_consistency_loss(logits, torch.randn(B, 14, 3),
                                  torch.zeros(B, dtype=torch.bool))
    assert none_.item() == 0.0, none_
    print("ok: view-consistency zero on identical logits, positive otherwise, "
          "zero when no study has auxiliaries")


def test_view_consistency_symmetric():
    a, b = torch.randn(B, 14, 3), torch.randn(B, 14, 3)
    has_aux = torch.ones(B, dtype=torch.bool)
    ab = view_consistency_loss(a, b, has_aux)
    ba = view_consistency_loss(b, a, has_aux)
    assert torch.allclose(ab, ba, atol=1e-6), (ab.item(), ba.item())
    print("ok: view-consistency is symmetric in its two arguments")


if __name__ == "__main__":
    test_mpc_zero_without_aux()
    test_mpc_rewards_same_study_similarity()
    test_mpc_partial_mask_is_finite()
    test_view_consistency_zero_when_identical()
    test_view_consistency_symmetric()
    print("\nall multi-view loss tests passed")
