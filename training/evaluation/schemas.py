"""Serialization schema for classification and generation predictions.

The whole point of this module is that **evaluation must be re-runnable without a
GPU and without the model**. Stage-1 evaluation currently throws probabilities
away immediately after ``argmax``, so recomputing any metric means re-running
inference over the whole split. A saved prediction file removes that coupling.

Design constraints:

* **numpy only.** No torch import at module level, so an ``.npz`` written on a
  GPU box can be read on a laptop with nothing but numpy installed.
* **Identifiers are opt-in.** ``subject_id``/``study_id``/``dicom_id`` are MIMIC
  identifiers under a PhysioNet DUA. They are stored only when the caller passes
  ``include_identifiers=True``; the default writes a salted digest instead.
* **Shapes are validated on construction**, not on first use. A mismatch between
  ``labels`` and ``probabilities`` is a programming error that should surface at
  the point of the bug, not three metrics later.

Label encoding matches ``ReportDataset``: ``0=negative, 1=positive,
2=uncertain``. ``-1`` marks a genuinely missing label; see
``training/evaluation/uncertain_policy.py`` for how each policy treats them.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

NEGATIVE = 0
POSITIVE = 1
UNCERTAIN = 2
MISSING = -1

#: Canonical class order, matching ``CLASS_MAP`` in
#: ``training/train_eval_figure9_llm_variants_200.py``.
CLASS_NAMES = ("negative", "positive", "uncertain")

#: Labels in the 14-column CheXpert vector that are not pathologies. ``No
#: Finding`` is a meta-label (the negation of the other thirteen) and ``Support
#: Devices`` is equipment. Averaging either into a macro pathology score is not
#: what the radiology literature reports, so they are excluded from macro
#: aggregates by default and still reported per-label.
META_LABELS = ("No Finding", "Support Devices")


class SchemaError(ValueError):
    """A prediction array does not satisfy the documented contract."""


def _digest(value: Any, salt: str) -> str:
    """Stable, non-reversible id for a record."""
    return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]


@dataclass
class ClassificationPredictions:
    """One split's worth of Stage-1 predictions.

    Attributes
    ----------
    labels:
        ``[N, P]`` int array of class indices. ``-1`` means the label is missing
        for that (sample, pathology) pair and every metric must skip it.
    probabilities:
        ``[N, P, C]`` float array. Rows are expected to sum to 1 along ``C``;
        this is checked but only warned about, since a caller may deliberately
        pass unnormalised scores.
    logits:
        Optional ``[N, P, C]`` float array, kept for exact reproducibility.
    sample_keys:
        ``[N]`` array of opaque per-study keys. Never a raw MIMIC identifier
        unless the caller explicitly opted in.
    """

    labels: np.ndarray
    probabilities: np.ndarray
    pathology_names: tuple[str, ...]
    sample_keys: np.ndarray
    logits: np.ndarray | None = None
    view_positions: np.ndarray | None = None
    num_views: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels)
        self.probabilities = np.asarray(self.probabilities, dtype=np.float64)

        if self.labels.ndim != 2:
            raise SchemaError(
                f"labels must be [N, P]; got shape {self.labels.shape}"
            )
        if self.probabilities.ndim != 3:
            raise SchemaError(
                f"probabilities must be [N, P, C]; got shape {self.probabilities.shape}"
            )
        if self.probabilities.shape[:2] != self.labels.shape:
            raise SchemaError(
                "probabilities[:2] must match labels: "
                f"{self.probabilities.shape[:2]} vs {self.labels.shape}"
            )
        if len(self.pathology_names) != self.labels.shape[1]:
            raise SchemaError(
                f"got {len(self.pathology_names)} pathology names for "
                f"{self.labels.shape[1]} label columns"
            )

        self.sample_keys = np.asarray(self.sample_keys)
        if self.sample_keys.shape[0] != self.labels.shape[0]:
            raise SchemaError(
                f"got {self.sample_keys.shape[0]} sample keys for "
                f"{self.labels.shape[0]} rows"
            )

        if self.logits is not None:
            self.logits = np.asarray(self.logits, dtype=np.float64)
            if self.logits.shape != self.probabilities.shape:
                raise SchemaError(
                    f"logits shape {self.logits.shape} != probabilities shape "
                    f"{self.probabilities.shape}"
                )

        bad = np.unique(self.labels[(self.labels < MISSING) | (self.labels >= self.num_classes)])
        if bad.size:
            raise SchemaError(
                f"labels contain out-of-range values {bad.tolist()}; expected "
                f"-1 (missing) or 0..{self.num_classes - 1}"
            )

        sums = self.probabilities.sum(axis=-1)
        if sums.size and not np.allclose(sums, 1.0, atol=1e-3):
            worst = float(np.max(np.abs(sums - 1.0)))
            logger.warning(
                "probabilities do not sum to 1 along the class axis "
                "(max deviation %.4g); AUROC/AUPRC use the positive column as-is",
                worst,
            )

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])

    @property
    def num_pathologies(self) -> int:
        return int(self.labels.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.probabilities.shape[2])

    @property
    def positive_probabilities(self) -> np.ndarray:
        """``[N, P]`` probability of the positive class.

        This is the score AUROC/AUPRC and threshold calibration operate on. It
        is deliberately the raw positive column rather than a renormalised
        positive-vs-rest quantity: renormalising would change the ranking when
        the uncertain mass differs between samples.
        """
        return self.probabilities[..., POSITIVE]

    def save(self, path: str | Path) -> Path:
        """Write a compressed ``.npz``. Returns the path actually written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "labels": self.labels,
            "probabilities": self.probabilities,
            "sample_keys": self.sample_keys.astype("U"),
            "pathology_names": np.asarray(self.pathology_names, dtype="U"),
        }
        if self.logits is not None:
            arrays["logits"] = self.logits
        if self.view_positions is not None:
            arrays["view_positions"] = np.asarray(self.view_positions, dtype="U")
        if self.num_views is not None:
            arrays["num_views"] = np.asarray(self.num_views)
        arrays["metadata_json"] = np.asarray(json.dumps(self.metadata, default=str))
        np.savez_compressed(path, **arrays)
        logger.info(
            "wrote %d predictions x %d pathologies to %s",
            self.num_samples,
            self.num_pathologies,
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ClassificationPredictions:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"prediction file not found: {path}")
        with np.load(path, allow_pickle=False) as handle:
            metadata: dict[str, Any] = {}
            if "metadata_json" in handle:
                metadata = json.loads(str(handle["metadata_json"]))
            return cls(
                labels=handle["labels"],
                probabilities=handle["probabilities"],
                pathology_names=tuple(str(x) for x in handle["pathology_names"]),
                sample_keys=handle["sample_keys"],
                logits=handle["logits"] if "logits" in handle else None,
                view_positions=(
                    handle["view_positions"] if "view_positions" in handle else None
                ),
                num_views=handle["num_views"] if "num_views" in handle else None,
                metadata=metadata,
            )

    def macro_pathology_indices(self, include_meta_labels: bool = False) -> list[int]:
        """Column indices that macro aggregates should average over."""
        if include_meta_labels:
            return list(range(self.num_pathologies))
        return [
            index
            for index, name in enumerate(self.pathology_names)
            if name not in META_LABELS
        ]


def build_sample_keys(
    identifiers: list[Any],
    *,
    include_identifiers: bool = False,
    salt: str = "meta-cxr-eval",
) -> np.ndarray:
    """Turn raw study identifiers into storable keys.

    With ``include_identifiers=False`` (the default) the result is a salted
    digest, so a prediction file can be shared without carrying MIMIC study
    identifiers.
    """
    if include_identifiers:
        return np.asarray([str(value) for value in identifiers], dtype="U")
    return np.asarray([_digest(value, salt) for value in identifiers], dtype="U")


@dataclass
class GenerationRecord:
    """One generated report paired with its reference."""

    sample_key: str
    generated: str
    reference: str
    view_position: str | None = None
    num_views: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record = {
            "sample_key": self.sample_key,
            "generated": self.generated,
            "reference": self.reference,
        }
        if self.view_position is not None:
            record["view_position"] = self.view_position
        if self.num_views is not None:
            record["num_views"] = self.num_views
        record.update(self.metadata)
        return record


def load_generation_records(path: str | Path) -> list[GenerationRecord]:
    """Read a JSONL file of generated reports.

    Duplicate ``sample_key`` values are an error rather than a silent
    last-one-wins: a duplicated study would be counted twice in every corpus
    metric and would break bootstrap resampling by study.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"generation file not found: {path}")

    records: list[GenerationRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(f"{path}:{line_number} is not valid JSON: {exc}") from exc

            missing = {"sample_key", "generated", "reference"} - set(payload)
            if missing:
                raise SchemaError(
                    f"{path}:{line_number} is missing required field(s): "
                    f"{', '.join(sorted(missing))}"
                )

            key = str(payload["sample_key"])
            if key in seen:
                raise SchemaError(
                    f"{path}:{line_number} repeats sample_key {key!r}; every "
                    "study must appear exactly once"
                )
            seen.add(key)

            known = {"sample_key", "generated", "reference", "view_position", "num_views"}
            records.append(
                GenerationRecord(
                    sample_key=key,
                    generated=str(payload["generated"]),
                    reference=str(payload["reference"]),
                    view_position=payload.get("view_position"),
                    num_views=payload.get("num_views"),
                    metadata={k: v for k, v in payload.items() if k not in known},
                )
            )

    if not records:
        raise SchemaError(f"{path} contains no records")
    logger.info("loaded %d generation records from %s", len(records), path)
    return records


def save_generation_records(
    records: list[GenerationRecord], path: str | Path
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path
