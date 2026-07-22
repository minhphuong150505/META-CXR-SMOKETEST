"""Assemble evaluation artifacts: JSON, CSV and a Markdown report.

Two rules govern everything written here.

**No invalid JSON.** ``float('nan')`` is not valid JSON, and ``json.dump``
happily writes a bare ``NaN`` token that most parsers reject. Every numeric
value passes through :func:`_json_safe`, which turns nan/inf into ``null``, and
the accompanying ``skipped`` section says *why* a value is null.

**No number without its provenance.** The metadata block records the git commit,
the checkpoint, the split, the seed, the uncertain policy, the threshold source
and the version of every metric package. A metric table that cannot be traced
back to a run is not usable in a thesis.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: Metrics whose headline value is worth a confidence interval.
HEADLINE_CLASSIFICATION_METRICS = (
    "positive_macro_f1",
    "positive_macro_recall",
    "positive_macro_precision",
    "macro_auroc",
    "macro_auprc",
    "positive_micro_f1",
)

HEADLINE_GENERATION_METRICS = (
    "bleu_4",
    "rouge_l",
    "meteor",
    "cider",
    "bertscore_f1",
)


def _json_safe(value: Any) -> Any:
    """Recursively replace nan/inf with None so the JSON stays parseable."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if (np.isnan(number) or np.isinf(number)) else number
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return bool(result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for package in (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "nltk",
        "bert-score",
        "pycocoevalcap",
        "matplotlib",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
        return "cpu"
    except ImportError:
        return "cpu (torch not installed)"


@dataclass
class ExperimentMetadata:
    """Everything needed to reproduce or cite a number."""

    split: str
    num_samples: int
    checkpoint: str = "unknown"
    config: str = "unknown"
    seed: int = 42
    uncertain_policy: str = "unknown"
    threshold_source: str = "unknown"
    num_pathologies: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        record = {
            "git_commit": _git_commit(),
            "git_dirty": _git_dirty(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "split": self.split,
            "num_samples": self.num_samples,
            "num_pathologies": self.num_pathologies,
            "checkpoint": self.checkpoint,
            "config": self.config,
            "seed": self.seed,
            "uncertain_policy": self.uncertain_policy,
            "threshold_source": self.threshold_source,
            "device": _device(),
            "platform": platform.platform(),
            "package_versions": _package_versions(),
        }
        record.update(self.extra)
        return record


def write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
    return path


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Write rows as CSV. ``nan`` becomes an empty cell, not the string 'nan'."""
    if not rows:
        raise ValueError(f"refusing to write an empty CSV to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for key, value in row.items():
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    cleaned[key] = ""
                elif isinstance(value, (list, dict)):
                    cleaned[key] = json.dumps(_json_safe(value), ensure_ascii=False)
                else:
                    cleaned[key] = value
            writer.writerow(cleaned)
    return path


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(_json_safe(row), ensure_ascii=False, allow_nan=False) + "\n"
            )
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return "n/a"
        return f"{value:.4f}"
    return str(value)


def _format_interval(interval: Any) -> str:
    """Render a CI from either a ConfidenceInterval or its ``to_dict`` form.

    The CLI serialises intervals before building the report, so both shapes
    reach this function.
    """
    if interval is None:
        return "-"
    if isinstance(interval, dict):
        lower, upper = interval.get("lower"), interval.get("upper")
    else:
        lower, upper = interval.lower, interval.upper
    if lower is None or upper is None:
        return "-"
    if isinstance(lower, float) and (np.isnan(lower) or np.isnan(upper)):
        return "-"
    return f"[{lower:.4f}, {upper:.4f}]"


def _metric_table(aggregates: dict[str, float], intervals: dict[str, Any]) -> str:
    lines = ["| Metric | Value | 95% CI |", "| --- | ---: | :---: |"]
    for name, value in aggregates.items():
        lines.append(
            f"| `{name}` | {_fmt(value)} | {_format_interval(intervals.get(name))} |"
        )
    return "\n".join(lines)


def build_markdown_report(
    metadata: ExperimentMetadata,
    *,
    classification: dict[str, Any] | None = None,
    generation: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> str:
    """Render ``evaluation_report.md``."""
    meta = metadata.to_dict()
    sections: list[str] = []

    sections.append(
        "# Evaluation report\n\n"
        f"Generated {meta['created_utc']} from commit `{meta['git_commit'][:12]}`"
        + (" (**working tree dirty**)" if meta["git_dirty"] else "")
        + "\n"
    )

    sections.append(
        "## Experiment metadata\n\n"
        "| Field | Value |\n| --- | --- |\n"
        + f"| Split | `{meta['split']}` |\n"
        + f"| Studies | {meta['num_samples']} |\n"
        + f"| Pathologies | {meta['num_pathologies']} |\n"
        + f"| Checkpoint | `{meta['checkpoint']}` |\n"
        + f"| Config | `{meta['config']}` |\n"
        + f"| Seed | {meta['seed']} |\n"
        + f"| Uncertain policy | `{meta['uncertain_policy']}` |\n"
        + f"| Threshold source | `{meta['threshold_source']}` |\n"
        + f"| Device | {meta['device']} |\n"
        + f"| Git commit | `{meta['git_commit']}` |\n"
    )

    versions = "\n".join(
        f"| `{name}` | {value} |" for name, value in meta["package_versions"].items()
    )
    sections.append(
        "### Metric package versions\n\n| Package | Version |\n| --- | --- |\n" + versions
    )

    if classification:
        sections.append(_classification_section(classification))
    if generation:
        sections.append(_generation_section(generation))

    limitation_lines = list(limitations or [])
    limitation_lines.extend(
        [
            "Lexical metrics (BLEU/ROUGE/METEOR/CIDEr/BERTScore) measure surface "
            "overlap. They do not measure clinical correctness: a report can score "
            "well while inverting a negation.",
            "All `possible_*` error flags are lexicon heuristics over surface text. "
            "They are screening signals for triage, **not** radiologist-confirmed "
            "clinical errors, and must not be reported as clinical error rates.",
            "Clinical metrics (CheXbert, RadGraph, RadCliQ, RadFact) are interfaced "
            "but not implemented in this repository. Any row showing them as "
            "unavailable means the dependency is absent, not that the model scored 0.",
            "No metric here substitutes for radiologist review.",
        ]
    )
    sections.append(
        "## Limitations\n\n" + "\n".join(f"- {line}" for line in limitation_lines)
    )

    return "\n\n---\n\n".join(sections) + "\n"


def _classification_section(payload: dict[str, Any]) -> str:
    parts = ["## Stage 1 — classification"]

    aggregates = payload.get("aggregates", {})
    intervals = payload.get("intervals", {})
    headline = {
        name: aggregates[name]
        for name in HEADLINE_CLASSIFICATION_METRICS
        if name in aggregates
    }
    parts.append("### Headline metrics\n\n" + _metric_table(headline, intervals))

    other = {k: v for k, v in aggregates.items() if k not in headline}
    if other:
        parts.append("### All aggregates\n\n" + _metric_table(other, {}))

    if payload.get("baseline_table"):
        parts.append(
            "### Baseline comparison\n\n"
            "A model whose accuracy is close to `all_negative` while its positive "
            "macro F1 is near 0 has not learned to detect findings.\n\n"
            + payload["baseline_table"]
        )

    rows = payload.get("per_pathology", [])
    if rows:
        header = (
            "| Pathology | n+ | Prev. | P | R | F1 | AUROC | AUPRC | Thr. |\n"
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        )
        body = "".join(
            f"| {row['pathology']} | {row['support_positive']} | "
            f"{_fmt(row['prevalence'])} | {_fmt(row['precision'])} | "
            f"{_fmt(row['recall'])} | {_fmt(row['f1'])} | {_fmt(row['auroc'])} | "
            f"{_fmt(row['auprc'])} | {_fmt(row['threshold'])} |\n"
            for row in rows
        )
        parts.append("### Per-pathology\n\n" + header + body)

    skipped = payload.get("skipped", {})
    if skipped:
        lines = "\n".join(
            f"- **{name}**: {', '.join(reasons)}" for name, reasons in sorted(skipped.items())
        )
        parts.append(
            "### Pathologies with undefined metrics\n\n"
            "These are **excluded** from the macro averages rather than counted "
            "as zero.\n\n" + lines
        )

    if payload.get("calibration_table"):
        parts.append("### Threshold calibration\n\n" + payload["calibration_table"])
    if payload.get("subgroup_table"):
        parts.append("### Subgroups\n\n" + payload["subgroup_table"])

    return "\n\n".join(parts)


def _generation_section(payload: dict[str, Any]) -> str:
    parts = ["## Stage 2 — report generation"]

    corpus = payload.get("corpus", {})
    intervals = payload.get("intervals", {})
    if corpus:
        parts.append("### Lexical metrics\n\n" + _metric_table(corpus, intervals))

    unavailable = payload.get("unavailable", {})
    if unavailable:
        lines = "\n".join(f"- **{name}**: {reason}" for name, reason in unavailable.items())
        parts.append(
            "### Unavailable metrics\n\n"
            "These are **not** reported as 0. The dependency is missing:\n\n" + lines
        )

    clinical = payload.get("clinical", {})
    if clinical:
        parts.append(
            "### Clinical metrics\n\n"
            + "\n".join(f"- **{name}**: {value}" for name, value in clinical.items())
        )

    errors = payload.get("errors", {})
    if errors:
        rates = errors.get("flag_rates", {})
        lines = "\n".join(
            f"| `{name}` | {errors['flag_counts'][name]} | {rate:.4f} |"
            for name, rate in rates.items()
        )
        parts.append(
            "### Error analysis\n\n"
            "| Flag | Count | Rate |\n| --- | ---: | ---: |\n" + lines
        )
        length = errors.get("generated_length", {})
        if length:
            parts.append(
                "### Report length (generated)\n\n"
                "| mean | min | p25 | median | p75 | max |\n"
                "| ---: | ---: | ---: | ---: | ---: | ---: |\n"
                f"| {length['mean']:.1f} | {length['min']:.0f} | {length['p25']:.0f} | "
                f"{length['median']:.0f} | {length['p75']:.0f} | {length['max']:.0f} |"
            )
        parts.append(f"> {errors.get('caveat', '')}")

    if payload.get("subgroup_table"):
        parts.append("### Subgroups\n\n" + payload["subgroup_table"])

    return "\n\n".join(parts)
