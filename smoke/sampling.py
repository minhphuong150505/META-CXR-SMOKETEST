"""Numerically safe sampling helpers used by the Stage-1 objectives."""

from __future__ import annotations

import torch


def negative_sampling_weights(
    similarity: torch.Tensor, candidate_mask: torch.Tensor
) -> torch.Tensor:
    """Softmax over eligible negatives without producing an empty distribution.

    Masking *after* softmax is unsafe: a very large positive-pair logit can make
    every negative probability underflow to zero, then removing the positive
    leaves ``torch.multinomial`` with a zero-sum row. Mask first so softmax is
    normalized over negatives themselves. Non-finite candidate scores use a
    deterministic finite subset, or a uniform eligible-candidate fallback when
    no finite score remains.
    """
    if similarity.ndim != 2 or candidate_mask.shape != similarity.shape:
        raise ValueError(
            "similarity and candidate_mask must be same-shaped rank-2 tensors; "
            f"got {tuple(similarity.shape)} and {tuple(candidate_mask.shape)}"
        )

    candidates = candidate_mask.to(device=similarity.device, dtype=torch.bool)
    candidate_counts = candidates.sum(dim=1, keepdim=True)
    if bool((candidate_counts == 0).any()):
        rows = (candidate_counts.squeeze(1) == 0).nonzero(as_tuple=True)[0].tolist()
        raise RuntimeError(f"negative sampling has no eligible candidate in rows {rows}")

    # Compute probabilities in fp32 even when the representation path is fp16.
    scores = similarity.float()
    positive_inf = candidates & torch.isposinf(scores)
    has_positive_inf = positive_inf.any(dim=1, keepdim=True)
    finite = candidates & torch.isfinite(scores)
    has_finite = finite.any(dim=1, keepdim=True)

    # If one or more eligible logits are +inf, their limiting softmax is uniform
    # over those maxima. Otherwise exclude NaN/-inf entries and softmax finite
    # candidates. Rows with no usable score are handled by the uniform fallback.
    inf_scores = torch.zeros_like(scores).masked_fill(~positive_inf, float("-inf"))
    finite_scores = scores.masked_fill(~finite, float("-inf"))
    safe_scores = torch.where(has_positive_inf, inf_scores, finite_scores)
    softmax_weights = torch.softmax(safe_scores, dim=1)

    fallback = candidates.float() / candidate_counts.float()
    has_usable_score = has_positive_inf | has_finite
    weights = torch.where(has_usable_score, softmax_weights, fallback)

    # Last defensive normalization keeps the contract explicit for multinomial.
    weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = weights.sum(dim=1, keepdim=True)
    weights = torch.where(row_sums > 0, weights / row_sums.clamp_min(1e-12), fallback)
    return weights
