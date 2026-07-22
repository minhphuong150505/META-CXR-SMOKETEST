"""Subgroup evaluation.

Only subgroups derivable from metadata the repository actually has are defined
here: **view position, view count, report length, and normal-vs-abnormal
studies**. There is deliberately no sex or age subgroup -- the processed split
CSVs carry no demographic column, and inventing one would mean joining
additional MIMIC metadata that this project has not been cleared to use for
that purpose.

Small subgroups are the trap. A subgroup of nine studies produces a positive
macro F1 that looks like a finding and is noise, so every subgroup row carries
its sample count and anything below :data:`MIN_RELIABLE_SAMPLES` is marked
unreliable in the output rather than quietly reported.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Below this, a subgroup metric is reported but flagged as unreliable.
MIN_RELIABLE_SAMPLES = 30

FRONTAL_VIEWS = ("PA", "AP")
LATERAL_VIEWS = ("LATERAL", "LL")


@dataclass
class Subgroup:
    """A named subset of sample indices."""

    name: str
    indices: np.ndarray
    description: str = ""

    @property
    def size(self) -> int:
        return int(self.indices.size)

    @property
    def reliable(self) -> bool:
        return self.size >= MIN_RELIABLE_SAMPLES


@dataclass
class SubgroupResult:
    """Metric values for one subgroup."""

    name: str
    size: int
    reliable: bool
    metrics: dict[str, float]
    description: str = ""
    warning: str = ""

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "subgroup": self.name,
            "samples": self.size,
            "reliable": self.reliable,
            "description": self.description,
        }
        if self.warning:
            record["warning"] = self.warning
        record.update(self.metrics)
        return record


def view_subgroups(
    view_positions: Sequence[str] | None, num_views: Sequence[int] | None
) -> list[Subgroup]:
    """Subgroups derived from view metadata.

    Returns an empty list when no view metadata is available, rather than
    fabricating a single "all" bucket that would imply the analysis ran.
    """
    subgroups: list[Subgroup] = []

    if view_positions is not None:
        views = np.asarray([str(v).strip().upper() for v in view_positions])
        for view in ("PA", "AP"):
            indices = np.where(views == view)[0]
            if indices.size:
                subgroups.append(
                    Subgroup(f"view_{view}", indices, f"anchor view is {view}")
                )
        lateral = np.where(np.isin(views, LATERAL_VIEWS))[0]
        if lateral.size:
            subgroups.append(
                Subgroup("view_lateral", lateral, "anchor view is LATERAL or LL")
            )

    if num_views is not None:
        counts = np.asarray(num_views, dtype=int)
        single = np.where(counts <= 1)[0]
        multi = np.where(counts > 1)[0]
        if single.size:
            subgroups.append(Subgroup("single_view", single, "study has one view"))
        if multi.size:
            subgroups.append(
                Subgroup("multi_view", multi, "study has more than one view")
            )

    return subgroups


def label_subgroups(
    labels: np.ndarray, pathology_names: Sequence[str], *, rare_threshold: float = 0.01
) -> list[Subgroup]:
    """Normal-vs-abnormal studies, split on whether any pathology is positive."""
    positive_any = np.any(labels == 1, axis=1)
    subgroups = []
    normal = np.where(~positive_any)[0]
    abnormal = np.where(positive_any)[0]
    if normal.size:
        subgroups.append(
            Subgroup("normal_studies", normal, "no pathology labelled positive")
        )
    if abnormal.size:
        subgroups.append(
            Subgroup(
                "abnormal_studies", abnormal, "at least one pathology labelled positive"
            )
        )
    return subgroups


def length_subgroups(
    lengths: Sequence[int], *, short_percentile: float = 25, long_percentile: float = 75
) -> list[Subgroup]:
    """Short vs long reference reports, split at the split's own quartiles."""
    values = np.asarray(lengths, dtype=float)
    if values.size == 0:
        return []
    short_cut = float(np.percentile(values, short_percentile))
    long_cut = float(np.percentile(values, long_percentile))

    short = np.where(values <= short_cut)[0]
    long = np.where(values >= long_cut)[0]
    subgroups = []
    if short.size:
        subgroups.append(
            Subgroup(
                "short_reports", short, f"reference length <= {short_cut:.0f} tokens"
            )
        )
    if long.size:
        subgroups.append(
            Subgroup("long_reports", long, f"reference length >= {long_cut:.0f} tokens")
        )
    return subgroups


def evaluate_subgroups(
    subgroups: list[Subgroup],
    metric_fn: Callable[[np.ndarray], dict[str, float]],
) -> list[SubgroupResult]:
    """Apply a metric function to each subgroup.

    ``metric_fn`` receives the subgroup's sample indices and returns a mapping of
    metric name to value. Exceptions are not swallowed: a subgroup that cannot be
    scored is a bug worth surfacing, not a row of zeros.
    """
    results: list[SubgroupResult] = []
    for subgroup in subgroups:
        metrics = metric_fn(subgroup.indices)
        warning = ""
        if not subgroup.reliable:
            warning = (
                f"only {subgroup.size} samples (< {MIN_RELIABLE_SAMPLES}); "
                "treat these values as indicative, not conclusive"
            )
            logger.warning("subgroup %s: %s", subgroup.name, warning)
        results.append(
            SubgroupResult(
                name=subgroup.name,
                size=subgroup.size,
                reliable=subgroup.reliable,
                metrics=metrics,
                description=subgroup.description,
                warning=warning,
            )
        )
    return results


def subgroup_table(results: list[SubgroupResult], columns: Sequence[str]) -> str:
    """Render subgroup results as a Markdown table."""
    header = "| Subgroup | Samples | " + " | ".join(columns) + " |\n"
    header += "| --- | ---: | " + " | ".join("---:" for _ in columns) + " |\n"

    body = ""
    for result in results:
        name = result.name if result.reliable else f"{result.name} ⚠️"
        values = []
        for column in columns:
            value = result.metrics.get(column)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                values.append("n/a")
            else:
                values.append(f"{value:.4f}")
        body += f"| {name} | {result.size} | " + " | ".join(values) + " |\n"

    if any(not r.reliable for r in results):
        body += (
            f"\n⚠️ = fewer than {MIN_RELIABLE_SAMPLES} samples; "
            "the metric is not reliable at that size.\n"
        )
    return header + body
