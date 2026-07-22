"""Bootstrap confidence intervals, resampled by study.

The resampling unit is the **study**, not the (sample, pathology) element. A
study contributes 14 correlated label decisions and one report; resampling those
independently would treat correlated observations as independent and produce
intervals that are far too narrow. This is the single most common way a
bootstrap CI is reported wrongly, so the unit is fixed here rather than being a
parameter.

Intervals are percentile intervals. BCa would be better for skewed statistics
but needs a jackknife pass per metric; percentile is the honest, standard choice
and is labelled as such in the output.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 42


class BootstrapError(ValueError):
    """Bootstrap was configured in a way that cannot produce a valid interval."""


@dataclass
class ConfidenceInterval:
    """A point estimate with a percentile bootstrap interval."""

    metric: str
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    samples: int
    seed: int
    valid_replicates: int
    method: str = "percentile"

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "point_estimate": _clean(self.point_estimate),
            "lower": _clean(self.lower),
            "upper": _clean(self.upper),
            "confidence": self.confidence,
            "bootstrap_samples": self.samples,
            "valid_replicates": self.valid_replicates,
            "seed": self.seed,
            "method": self.method,
        }

    def format(self) -> str:
        if np.isnan(self.point_estimate):
            return "n/a"
        return f"{self.point_estimate:.4f} [{self.lower:.4f}, {self.upper:.4f}]"


def _clean(value: float) -> float | None:
    """JSON has no NaN. Emit ``null`` so the file stays parseable."""
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def bootstrap_metric(
    metric_fn: Callable[[np.ndarray], float],
    num_studies: int,
    *,
    metric_name: str = "metric",
    samples: int = DEFAULT_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> ConfidenceInterval:
    """Bootstrap one scalar metric by resampling study indices.

    Parameters
    ----------
    metric_fn:
        Called with an array of study indices (with replacement) and must return
        a scalar. Returning ``nan`` marks the replicate as unusable -- for
        example when a resample happens to contain no positive samples -- and it
        is excluded rather than counted as zero.
    num_studies:
        Population size to resample from.
    """
    if samples < 0:
        raise BootstrapError(f"samples must be >= 0, got {samples}")
    if not 0 < confidence < 1:
        raise BootstrapError(f"confidence must be in (0, 1), got {confidence}")
    if num_studies == 0:
        raise BootstrapError("cannot bootstrap an empty split")

    all_indices = np.arange(num_studies)
    point_estimate = float(metric_fn(all_indices))

    if samples == 0:
        return ConfidenceInterval(
            metric=metric_name,
            point_estimate=point_estimate,
            lower=float("nan"),
            upper=float("nan"),
            confidence=confidence,
            samples=0,
            seed=seed,
            valid_replicates=0,
            method="disabled",
        )

    rng = np.random.default_rng(seed)
    replicates: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, num_studies, size=num_studies)
        value = float(metric_fn(indices))
        if not np.isnan(value):
            replicates.append(value)

    if not replicates:
        logger.warning(
            "every bootstrap replicate for %s was undefined; reporting no interval",
            metric_name,
        )
        return ConfidenceInterval(
            metric=metric_name,
            point_estimate=point_estimate,
            lower=float("nan"),
            upper=float("nan"),
            confidence=confidence,
            samples=samples,
            seed=seed,
            valid_replicates=0,
        )

    if len(replicates) < samples:
        logger.warning(
            "%d/%d bootstrap replicates for %s were undefined and were dropped",
            samples - len(replicates),
            samples,
            metric_name,
        )

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(replicates, 100 * alpha))
    upper = float(np.percentile(replicates, 100 * (1.0 - alpha)))

    return ConfidenceInterval(
        metric=metric_name,
        point_estimate=point_estimate,
        lower=lower,
        upper=upper,
        confidence=confidence,
        samples=samples,
        seed=seed,
        valid_replicates=len(replicates),
    )


def bootstrap_many(
    metric_fns: dict[str, Callable[[np.ndarray], float]],
    num_studies: int,
    *,
    samples: int = DEFAULT_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> dict[str, ConfidenceInterval]:
    """Bootstrap several metrics.

    Each metric gets the **same** seed, so all metrics are evaluated on the same
    sequence of resamples. That makes their intervals directly comparable and
    keeps the whole run reproducible from one seed.
    """
    return {
        name: bootstrap_metric(
            fn,
            num_studies,
            metric_name=name,
            samples=samples,
            confidence=confidence,
            seed=seed,
        )
        for name, fn in metric_fns.items()
    }


def bootstrap_sample_metric(
    per_sample_values: Sequence[float],
    *,
    metric_name: str = "metric",
    samples: int = DEFAULT_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> ConfidenceInterval:
    """Bootstrap the mean of a per-sample score (e.g. per-report ROUGE-L)."""
    values = np.asarray(per_sample_values, dtype=np.float64)
    if values.size == 0:
        raise BootstrapError(f"no per-sample values supplied for {metric_name}")

    def mean_of(indices: np.ndarray) -> float:
        selected = values[indices]
        if np.all(np.isnan(selected)):
            return float("nan")
        return float(np.nanmean(selected))

    return bootstrap_metric(
        mean_of,
        values.size,
        metric_name=metric_name,
        samples=samples,
        confidence=confidence,
        seed=seed,
    )
