"""Plots for Stage-1 evaluation.

matplotlib only, ``Agg`` backend, no seaborn: the evaluator has to run headless
on a training VM. Every function returns the path it wrote, or ``None`` when the
plot was impossible (a pathology with one class has no ROC curve). A missing
class is never a crash and never an empty axis presented as a result.

Plotting is optional. ``--no-plots`` skips this module entirely, which matters
because rendering 14 pathologies x 5 figures dominates the runtime of an
otherwise instant evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PlottingUnavailable(RuntimeError):
    """matplotlib is not installed."""


def _pyplot():
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise PlottingUnavailable(
            "matplotlib is required for plots. Install it, or pass --no-plots."
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _roc_points(scores: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    truth = y_true[order].astype(bool)
    tp = np.cumsum(truth)
    fp = np.cumsum(~truth)
    n_pos = int(truth.sum())
    n_neg = int(truth.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.array([]), np.array([])
    return np.r_[0, fp / n_neg, 1], np.r_[0, tp / n_pos, 1]


def _pr_points(scores: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    truth = y_true[order].astype(bool)
    tp = np.cumsum(truth)
    fp = np.cumsum(~truth)
    n_pos = int(truth.sum())
    if n_pos == 0:
        return np.array([]), np.array([])
    return tp / n_pos, tp / np.maximum(tp + fp, 1)


def plot_roc_curves(
    scores: np.ndarray,
    y_true: np.ndarray,
    valid: np.ndarray,
    pathology_names: tuple[str, ...],
    output_dir: Path,
) -> Path | None:
    """One figure with every pathology's ROC curve overlaid."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 7))
    drawn = 0
    for index, name in enumerate(pathology_names):
        column = valid[:, index]
        fpr, tpr = _roc_points(scores[column, index], y_true[column, index])
        if fpr.size == 0:
            logger.info("skipping ROC for %s: only one class present", name)
            continue
        axis.plot(fpr, tpr, linewidth=1.2, label=name)
        drawn += 1

    if drawn == 0:
        plt.close(figure)
        logger.warning("no ROC curve could be drawn; every pathology has one class")
        return None

    axis.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("ROC curves by pathology")
    axis.legend(fontsize=7, loc="lower right")
    figure.tight_layout()

    path = output_dir / "roc_curves.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_pr_curves(
    scores: np.ndarray,
    y_true: np.ndarray,
    valid: np.ndarray,
    pathology_names: tuple[str, ...],
    output_dir: Path,
) -> Path | None:
    """Precision-recall curves, with each pathology's prevalence as its floor."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(8, 7))
    drawn = 0
    for index, name in enumerate(pathology_names):
        column = valid[:, index]
        recall, precision = _pr_points(scores[column, index], y_true[column, index])
        if recall.size == 0:
            logger.info("skipping PR curve for %s: no positive samples", name)
            continue
        axis.plot(recall, precision, linewidth=1.2, label=name)
        drawn += 1

    if drawn == 0:
        plt.close(figure)
        return None

    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curves by pathology")
    axis.legend(fontsize=7, loc="upper right")
    figure.tight_layout()

    path = output_dir / "pr_curves.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_confusion_matrices(
    matrices: np.ndarray,
    pathology_names: tuple[str, ...],
    class_names: tuple[str, ...],
    output_dir: Path,
) -> list[Path]:
    """One heatmap per pathology, written into ``confusion_matrices/``."""
    plt = _pyplot()
    target = output_dir / "confusion_matrices"
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for index, name in enumerate(pathology_names):
        matrix = matrices[index]
        figure, axis = plt.subplots(figsize=(4.5, 4))
        axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
        axis.set_yticks(range(len(class_names)), class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(name, fontsize=10)

        maximum = matrix.max() if matrix.size else 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                axis.text(
                    j,
                    i,
                    f"{int(matrix[i, j])}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > maximum / 2 else "black",
                    fontsize=9,
                )
        figure.tight_layout()
        safe = name.replace(" ", "_").replace("/", "_")
        path = target / f"confusion_{safe}.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)
    return written


def plot_per_pathology_bars(
    values: dict[str, float],
    title: str,
    ylabel: str,
    output_dir: Path,
    filename: str,
) -> Path:
    """Bar chart over pathologies. Undefined values are drawn as a gap."""
    plt = _pyplot()
    names = list(values)
    heights = [0.0 if np.isnan(values[n]) else values[n] for n in names]
    undefined = [np.isnan(values[n]) for n in names]

    figure, axis = plt.subplots(figsize=(max(6, len(names) * 0.6), 4.5))
    bars = axis.bar(range(len(names)), heights, color="#4C72B0")
    for bar, is_undefined in zip(bars, undefined, strict=True):
        if is_undefined:
            bar.set_color("#CCCCCC")
            bar.set_hatch("//")

    axis.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    if any(undefined):
        axis.text(
            0.99,
            0.97,
            "hatched = undefined",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#666666",
        )
    figure.tight_layout()

    path = output_dir / filename
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_threshold_comparison(
    default_scores: dict[str, float],
    calibrated_scores: dict[str, float],
    output_dir: Path,
) -> Path:
    """Side-by-side F1 at threshold 0.5 versus at the calibrated threshold."""
    plt = _pyplot()
    names = list(default_scores)
    x = np.arange(len(names))
    width = 0.4

    figure, axis = plt.subplots(figsize=(max(6, len(names) * 0.7), 4.5))
    axis.bar(
        x - width / 2,
        [0.0 if np.isnan(default_scores[n]) else default_scores[n] for n in names],
        width,
        label="threshold 0.5",
        color="#CCCCCC",
    )
    axis.bar(
        x + width / 2,
        [
            0.0 if np.isnan(calibrated_scores.get(n, np.nan)) else calibrated_scores[n]
            for n in names
        ],
        width,
        label="calibrated",
        color="#4C72B0",
    )
    axis.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
    axis.set_ylabel("Positive-class F1")
    axis.set_title("Default vs calibrated threshold")
    axis.legend()
    figure.tight_layout()

    path = output_dir / "threshold_comparison.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_probability_histogram(
    scores: np.ndarray,
    y_true: np.ndarray,
    valid: np.ndarray,
    pathology_names: tuple[str, ...],
    output_dir: Path,
) -> Path | None:
    """Positive-probability distribution for positive vs negative samples.

    A model that has learned something separates the two histograms. Complete
    overlap is visible here long before it shows up in a summary table.
    """
    plt = _pyplot()
    positive_scores = scores[valid & (y_true == 1)]
    negative_scores = scores[valid & (y_true == 0)]
    if positive_scores.size == 0 or negative_scores.size == 0:
        logger.warning("cannot draw probability histogram: one class is empty")
        return None

    figure, axis = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0, 1, 41)
    axis.hist(negative_scores, bins=bins, alpha=0.6, label="negative", density=True)
    axis.hist(positive_scores, bins=bins, alpha=0.6, label="positive", density=True)
    axis.set_xlabel("Predicted probability of positive")
    axis.set_ylabel("Density")
    axis.set_title("Positive-class probability by true label (all pathologies pooled)")
    axis.legend()
    figure.tight_layout()

    path = output_dir / "probability_histogram.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_reliability_diagram(
    scores: np.ndarray,
    y_true: np.ndarray,
    valid: np.ndarray,
    output_dir: Path,
    bins: int = 10,
) -> Path | None:
    """Calibration curve: predicted probability vs observed frequency."""
    plt = _pyplot()
    flat_scores = scores[valid]
    flat_true = y_true[valid].astype(float)
    if flat_scores.size == 0:
        return None

    edges = np.linspace(0, 1, bins + 1)
    centres, observed = [], []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (flat_scores >= low) & (flat_scores < high)
        if not np.any(in_bin):
            continue
        centres.append(float(np.mean(flat_scores[in_bin])))
        observed.append(float(np.mean(flat_true[in_bin])))

    if not centres:
        return None

    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfectly calibrated")
    axis.plot(centres, observed, "o-", label="model")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed positive frequency")
    axis.set_title("Reliability diagram")
    axis.legend()
    figure.tight_layout()

    path = output_dir / "reliability_diagram.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
