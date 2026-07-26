"""The MHCAC contrastive term and its fp16 failure mode.

Run e123 reported a `loss_contrastive` far above what the formula can produce,
alongside `grad_norm=nan`, `amp_overflow_steps>0` and `optimizer_steps_taken=0`.
This pins the two facts that follow from the code:

  1. ``AbnormalitySpecificLoss`` is analytically bounded by ``margin + 2`` --
     both of its terms are computed on L2-normalised tokens. A *finite* value
     above that bound cannot come from the formula, so it now raises instead of
     being handed to backward() where it resurfaces as an exploding grad norm.

  2. The bookkeeping zero the module returns from its no-op branches used to be
     ``common_representations.sum() * 0.0``. Under fp16 autocast that ``sum``
     saturates to ``inf`` on a batch of realistic size, and ``inf * 0.0`` is
     ``nan`` -- a value that has nothing to do with the data, poisons the total
     loss, and gets every step dropped by the non-finite guard in
     ``BaseTask._train_inner_loop``. That is the mechanism by which a run can
     complete two epochs having applied zero optimizer updates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mhcac.loss import AbnormalitySpecificLoss, graph_connected_zero  # noqa: E402

D_EMBED = 32
NUM_ABNORMALITIES = 14


def _module(margin=0.7):
    torch.manual_seed(0)
    return AbnormalitySpecificLoss(
        margin=margin, d_embedding=D_EMBED, num_abnormalities=NUM_ABNORMALITIES
    )


def _inputs(batch=8, num_tokens=NUM_ABNORMALITIES, scale=1.0):
    torch.manual_seed(1)
    reps = torch.randn(batch, num_tokens, D_EMBED) * scale
    attn = [torch.softmax(torch.randn(batch, num_tokens, 16), dim=-1) for _ in range(3)]
    return reps, attn


# --------------------------------------------------------------------------- #
# 1. The analytic bound                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", [1.0, 50.0, 1000.0])
def test_contrastive_term_stays_inside_its_analytic_bound(scale):
    """pos_neg <= margin (relu of a quantity <= margin) and unc <= 2 (a
    difference of two cosine means), so the mean over abnormalities <= margin+2
    for any input magnitude."""
    module = _module()
    reps, attn = _inputs(scale=scale)
    labels = torch.randint(0, 3, (reps.shape[0], NUM_ABNORMALITIES))

    _, _, contrastive, _ = module(reps, attn, labels)

    bound = module.margin + AbnormalitySpecificLoss.MAX_UNCERTAIN_TERM
    assert torch.isfinite(contrastive)
    assert 0.0 <= float(contrastive.detach()) <= bound


def test_out_of_bound_value_raises_instead_of_reaching_backward():
    """Exercises the guard itself.

    Real inputs cannot breach the bound (that is the point of the test above),
    so the bound is lowered to make the module's own output out-of-range. What
    this pins is the reaction: raise, naming the value and the bound, rather
    than pass the number to backward() where it resurfaces much later as an
    unexplained non-finite grad norm.
    """
    module = _module()
    module.MAX_UNCERTAIN_TERM = -1.0  # bound becomes 0.7 - 1.0 < 0
    reps, attn = _inputs()
    labels = torch.randint(0, 3, (reps.shape[0], NUM_ABNORMALITIES))

    with pytest.raises(RuntimeError, match="exceeds its analytic bound"):
        module(reps, attn, labels)


# --------------------------------------------------------------------------- #
# 2. The fp16 zero                                                             #
# --------------------------------------------------------------------------- #
def test_naive_zero_touch_overflows_in_fp16():
    """Documents the bug being fixed: the old spelling produces nan, not zero,
    on a tensor whose fp16 sum exceeds 65504."""
    big = torch.full((8, 14, 768), 0.9, dtype=torch.float16)
    assert torch.isinf(big.sum())
    assert torch.isnan(big.sum() * 0.0)


def test_graph_connected_zero_is_zero_and_keeps_the_graph_edge():
    big = torch.full((8, 14, 768), 0.9, dtype=torch.float16, requires_grad=True)

    zero = graph_connected_zero(big)

    assert float(zero.detach()) == 0.0
    assert zero.requires_grad, "the zero must still carry the tensor into autograd"
    zero.backward()
    assert big.grad is not None and torch.all(big.grad == 0)


@pytest.mark.parametrize(
    "labels_missing",
    [True, False],
    ids=["labels_none", "empty_sample_mask"],
)
def test_noop_branches_return_a_real_zero_in_fp16(labels_missing):
    """The two branches that have nothing to contribute must return 0.0, not
    nan. Returning nan here is what made the total loss non-finite on batches
    that were perfectly valid apart from having no usable classification rows."""
    module = _module().half()
    reps, attn = _inputs()
    reps = (reps.half() * 0 + 0.9).requires_grad_(True)
    attn = [a.half() for a in attn]

    if labels_missing:
        _, _, contrastive, _ = module(reps, attn, None)
    else:
        labels = torch.randint(0, 3, (reps.shape[0], NUM_ABNORMALITIES))
        mask = torch.zeros(reps.shape[0], dtype=torch.bool)
        _, _, contrastive, _ = module(reps, attn, labels, sample_mask=mask)

    assert torch.isfinite(contrastive), "no-op branch produced a non-finite loss"
    assert float(contrastive.detach()) == 0.0


# --------------------------------------------------------------------------- #
# 3. The aliased accumulator (the actual run-e123 root cause)                   #
# --------------------------------------------------------------------------- #
def test_skipped_gates_do_not_amplify_the_accumulated_loss():
    """`contrastive_loss` starts out as the *same tensor object* as `zero`, and
    `zero` is the fallback both branches use when an abnormality has no
    positives / no negatives / no uncertains in the batch. With an in-place
    `+=` that made `zero` become the running total, so each later skipped gate
    added the accumulated loss back into itself -- c -> 2c or 3c, i.e. 2**k
    growth over 14 abnormalities. That is where 387 / 827 / 7040 came from.

    Constructed so most abnormalities skip at least one gate: single-class
    columns leave pos_indices or neg_indices empty.
    """
    module = _module()
    reps, attn = _inputs()
    labels = torch.zeros(reps.shape[0], NUM_ABNORMALITIES, dtype=torch.long)
    # Only column 0 has all three classes; the other 13 are single-class, so
    # both gates skip there -> 3x per column on the old in-place code.
    labels[:, 0] = torch.tensor([1, 0, 2, 1, 1, 0, 2, 0])

    _, _, contrastive, _ = module(reps, attn, labels)

    bound = module.margin + AbnormalitySpecificLoss.MAX_UNCERTAIN_TERM
    assert float(contrastive.detach()) <= bound


def test_in_place_accumulation_onto_an_aliased_zero_is_the_amplifier():
    """The mechanism itself, in four lines: this is what the fix removes."""
    zero = torch.zeros(())
    total = zero  # same object, exactly as `contrastive_loss = zero` was

    total += torch.tensor(0.02)  # one abnormality contributes
    assert float(zero) == pytest.approx(0.02), "in-place mutated the shared zero"

    total += zero + zero  # a later abnormality skips BOTH gates -> 3x
    assert float(total) == pytest.approx(0.06)

    # Out-of-place leaves the fallback pristine, so a skipped gate adds nothing.
    zero = torch.zeros(())
    total = zero
    total = total + torch.tensor(0.02)
    total = total + (zero + zero)
    assert float(zero) == 0.0
    assert float(total) == pytest.approx(0.02)
