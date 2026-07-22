"""How the uncertain class is folded into binary positive-vs-rest metrics.

CheXpert's uncertain label (raw ``-1``, stored as class ``2``) has no single
correct treatment: the literature reports U-Ones, U-Zeros and U-Ignore, and they
give materially different numbers. Making the choice a silent constant would
make results incomparable, so it is an explicit parameter that is recorded in
every output file.

Policies
--------
``three_class``
    Keep negative/positive/uncertain as three distinct classes. Binary
    positive-vs-rest metrics treat uncertain as *not* positive, which is the
    behaviour of the existing ``argmax``-based evaluator.
``uncertain_as_positive``
    U-Ones. Uncertain counts as a positive finding.
``uncertain_as_negative``
    U-Zeros. Uncertain counts as a negative finding.
``ignore_uncertain``
    U-Ignore. Uncertain (sample, pathology) pairs are dropped from binary
    metrics entirely -- they contribute to neither numerator nor denominator.

``three_class`` is the default because it reproduces the current pipeline's
behaviour. Note it is *not* the same as ``uncertain_as_negative`` for
three-class metrics (confusion matrices keep three rows), but the two agree on
every binary positive-vs-rest metric.
"""

from __future__ import annotations

import numpy as np

from training.evaluation.schemas import MISSING, POSITIVE, UNCERTAIN

THREE_CLASS = "three_class"
UNCERTAIN_AS_POSITIVE = "uncertain_as_positive"
UNCERTAIN_AS_NEGATIVE = "uncertain_as_negative"
IGNORE_UNCERTAIN = "ignore_uncertain"

POLICIES = (
    THREE_CLASS,
    UNCERTAIN_AS_POSITIVE,
    UNCERTAIN_AS_NEGATIVE,
    IGNORE_UNCERTAIN,
)

DEFAULT_POLICY = THREE_CLASS


class UnknownPolicyError(ValueError):
    """The requested uncertain-label policy does not exist."""


def validate_policy(policy: str) -> str:
    if policy not in POLICIES:
        raise UnknownPolicyError(
            f"unknown uncertain policy {policy!r}; expected one of "
            f"{', '.join(POLICIES)}"
        )
    return policy


def binarize_labels(
    labels: np.ndarray, policy: str = DEFAULT_POLICY
) -> tuple[np.ndarray, np.ndarray]:
    """Map class-index labels to binary positive-vs-rest, plus a validity mask.

    Parameters
    ----------
    labels:
        ``[N, P]`` int array of class indices, where ``-1`` marks a missing
        label.
    policy:
        One of :data:`POLICIES`.

    Returns
    -------
    binary:
        ``[N, P]`` int array in ``{0, 1}``. Entries where ``valid`` is False are
        set to 0 and must not be read.
    valid:
        ``[N, P]`` bool array. False wherever the label is missing, or the label
        is uncertain under ``ignore_uncertain``.

    Every downstream binary metric consumes this pair. Returning the mask
    separately -- rather than encoding "drop me" as a magic label value -- is
    what keeps ``ignore_uncertain`` from being silently reinterpreted as a
    negative somewhere further down.
    """
    validate_policy(policy)
    labels = np.asarray(labels)

    valid = labels != MISSING
    binary = np.zeros_like(labels, dtype=np.int64)

    is_positive = labels == POSITIVE
    is_uncertain = labels == UNCERTAIN

    if policy == UNCERTAIN_AS_POSITIVE:
        binary[is_positive | is_uncertain] = 1
    elif policy == IGNORE_UNCERTAIN:
        binary[is_positive] = 1
        valid = valid & ~is_uncertain
    else:
        # three_class and uncertain_as_negative agree here: only an explicit
        # positive counts as positive.
        binary[is_positive] = 1

    binary = np.where(valid, binary, 0)
    return binary, valid


def describe_policy(policy: str) -> str:
    """One-line human description, for report headers."""
    validate_policy(policy)
    return {
        THREE_CLASS: (
            "uncertain kept as its own class; not counted as positive in "
            "binary metrics"
        ),
        UNCERTAIN_AS_POSITIVE: "uncertain merged into positive (U-Ones)",
        UNCERTAIN_AS_NEGATIVE: "uncertain merged into negative (U-Zeros)",
        IGNORE_UNCERTAIN: "uncertain dropped from binary metrics (U-Ignore)",
    }[policy]


def label_counts(labels: np.ndarray) -> dict[str, int]:
    """Raw class counts, before any policy is applied."""
    labels = np.asarray(labels)
    return {
        "negative": int(np.sum(labels == 0)),
        "positive": int(np.sum(labels == POSITIVE)),
        "uncertain": int(np.sum(labels == UNCERTAIN)),
        "missing": int(np.sum(labels == MISSING)),
    }
