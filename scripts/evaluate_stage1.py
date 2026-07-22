#!/usr/bin/env python3
"""Evaluate Stage-1 classification from a saved prediction file.

No model, no GPU and no dataset are needed: everything is recomputed from the
``.npz`` written during inference. That is the point -- re-running an evaluation
with a different uncertain policy or a new threshold file must not cost a
GPU-hour.

    python scripts/evaluate_stage1.py \\
        --predictions outputs/test_predictions.npz \\
        --thresholds outputs/thresholds.json \\
        --uncertain-policy ignore_uncertain \\
        --bootstrap-samples 1000 \\
        --output-dir outputs/stage1_evaluation

Exit codes: 0 success, 1 evaluation failed, 2 bad arguments or input.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.baselines import baseline_table, compute_baselines  # noqa: E402
from training.evaluation.bootstrap import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_SAMPLES,
    DEFAULT_SEED,
    bootstrap_many,
)
from training.evaluation.classification_metrics import evaluate_classification  # noqa: E402
from training.evaluation.report_writer import (  # noqa: E402
    ExperimentMetadata,
    build_markdown_report,
    write_csv,
    write_json,
)
from training.evaluation.schemas import CLASS_NAMES, ClassificationPredictions  # noqa: E402
from training.evaluation.subgroup_analysis import (  # noqa: E402
    evaluate_subgroups,
    label_subgroups,
    subgroup_table,
    view_subgroups,
)
from training.evaluation.threshold_calibration import (  # noqa: E402
    CalibrationError,
    load_thresholds,
)
from training.evaluation.uncertain_policy import (  # noqa: E402
    DEFAULT_POLICY,
    POLICIES,
    binarize_labels,
)

logger = logging.getLogger("evaluate_stage1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--predictions", required=True, type=Path, help="Prediction .npz to evaluate."
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        help="Calibrated thresholds from scripts/calibrate_thresholds.py. "
        "Omit to use 0.5 for every pathology.",
    )
    parser.add_argument(
        "--uncertain-policy", default=DEFAULT_POLICY, choices=sorted(POLICIES)
    )
    parser.add_argument(
        "--include-meta-labels",
        action="store_true",
        help="Include 'No Finding' and 'Support Devices' in macro averages "
        "(they are excluded by default; both are always reported per-pathology).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--bootstrap-confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--evaluation-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap CIs.")
    parser.add_argument("--no-plots", action="store_true", help="Skip all figures.")
    parser.add_argument("--no-baselines", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", default=None, help="Override the recorded split name.")
    parser.add_argument("--checkpoint", default="unknown")
    parser.add_argument("--config", default="unknown")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not args.predictions.is_file():
        logger.error("prediction file not found: %s", args.predictions)
        return 2

    try:
        predictions = ClassificationPredictions.load(args.predictions)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        logger.error("could not read %s: %s", args.predictions, exc)
        return 2

    if predictions.num_samples == 0:
        logger.error("%s contains no samples", args.predictions)
        return 2

    logger.info(
        "loaded %d studies x %d pathologies from %s",
        predictions.num_samples,
        predictions.num_pathologies,
        args.predictions,
    )

    thresholds = None
    threshold_source = "default 0.5 for every pathology"
    if args.thresholds:
        try:
            thresholds = load_thresholds(args.thresholds)
        except (CalibrationError, FileNotFoundError) as exc:
            logger.error("%s", exc)
            return 2
        threshold_source = str(args.thresholds)
        logger.info("loaded %d calibrated thresholds", len(thresholds))

    report = evaluate_classification(
        predictions,
        thresholds=thresholds,
        uncertain_policy=args.uncertain_policy,
        include_meta_labels=args.include_meta_labels,
    )

    if report.skipped:
        for name, reasons in sorted(report.skipped.items()):
            logger.info("pathology %s excluded where undefined: %s", name, ", ".join(reasons))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "aggregates": report.aggregates,
        "per_pathology": [m.to_dict() for m in report.per_pathology],
        "skipped": report.skipped,
        "settings": report.settings,
        "macro_pathologies": report.macro_pathologies,
    }

    # ---- bootstrap -------------------------------------------------------
    intervals = {}
    if not args.no_bootstrap and args.bootstrap_samples > 0:
        logger.info(
            "bootstrapping %d replicates by study (seed=%d)",
            args.bootstrap_samples,
            args.evaluation_seed,
        )

        def metric_getter(name: str):
            def compute(indices: np.ndarray) -> float:
                subset = ClassificationPredictions(
                    labels=predictions.labels[indices],
                    probabilities=predictions.probabilities[indices],
                    pathology_names=predictions.pathology_names,
                    sample_keys=predictions.sample_keys[indices],
                )
                return evaluate_classification(
                    subset,
                    thresholds=thresholds,
                    uncertain_policy=args.uncertain_policy,
                    include_meta_labels=args.include_meta_labels,
                ).aggregates[name]

            return compute

        from training.evaluation.report_writer import HEADLINE_CLASSIFICATION_METRICS

        intervals = bootstrap_many(
            {name: metric_getter(name) for name in HEADLINE_CLASSIFICATION_METRICS},
            predictions.num_samples,
            samples=args.bootstrap_samples,
            confidence=args.bootstrap_confidence,
            seed=args.evaluation_seed,
        )
        payload["intervals"] = {k: v.to_dict() for k, v in intervals.items()}

    # ---- baselines -------------------------------------------------------
    if not args.no_baselines:
        rows = compute_baselines(
            predictions,
            uncertain_policy=args.uncertain_policy,
            include_meta_labels=args.include_meta_labels,
            seed=args.evaluation_seed,
        )
        from training.evaluation.baselines import BaselineRow

        model_row = BaselineRow(
            name="model",
            accuracy=report.aggregates["binary_accuracy"],
            positive_macro_f1=report.aggregates["positive_macro_f1"],
            positive_macro_recall=report.aggregates["positive_macro_recall"],
            macro_auroc=report.aggregates["macro_auroc"],
            macro_auprc=report.aggregates["macro_auprc"],
            description="the evaluated model",
        )
        payload["baselines"] = [row.to_dict() for row in rows]
        payload["baseline_table"] = baseline_table(rows, model_row)

    # ---- subgroups -------------------------------------------------------
    subgroups = view_subgroups(predictions.view_positions, predictions.num_views)
    subgroups.extend(label_subgroups(predictions.labels, predictions.pathology_names))
    if subgroups:

        def subgroup_metrics(indices: np.ndarray) -> dict[str, float]:
            subset = ClassificationPredictions(
                labels=predictions.labels[indices],
                probabilities=predictions.probabilities[indices],
                pathology_names=predictions.pathology_names,
                sample_keys=predictions.sample_keys[indices],
            )
            aggregates = evaluate_classification(
                subset,
                thresholds=thresholds,
                uncertain_policy=args.uncertain_policy,
                include_meta_labels=args.include_meta_labels,
            ).aggregates
            return {
                "positive_macro_f1": aggregates["positive_macro_f1"],
                "macro_auroc": aggregates["macro_auroc"],
                "macro_auprc": aggregates["macro_auprc"],
            }

        results = evaluate_subgroups(subgroups, subgroup_metrics)
        payload["subgroups"] = [r.to_dict() for r in results]
        payload["subgroup_table"] = subgroup_table(
            results, ["positive_macro_f1", "macro_auroc", "macro_auprc"]
        )

    # ---- plots -----------------------------------------------------------
    if not args.no_plots:
        try:
            _write_plots(predictions, report, args, output_dir)
        except Exception as exc:  # noqa: BLE001
            # A plotting failure must not lose the metrics that already computed.
            logger.warning("plotting failed (metrics are unaffected): %s", exc)

    # ---- artifacts -------------------------------------------------------
    metadata = ExperimentMetadata(
        split=args.split or str(predictions.metadata.get("split", "unknown")),
        num_samples=predictions.num_samples,
        num_pathologies=predictions.num_pathologies,
        checkpoint=args.checkpoint or str(predictions.metadata.get("checkpoint", "unknown")),
        config=args.config,
        seed=args.evaluation_seed,
        uncertain_policy=args.uncertain_policy,
        threshold_source=threshold_source,
    )

    write_json(
        {"metadata": metadata.to_dict(), "classification": payload},
        output_dir / "metrics.json",
    )
    write_csv([m.to_dict() for m in report.per_pathology], output_dir / "per_pathology_metrics.csv")
    write_csv(
        [{"metric": k, "value": v} for k, v in report.aggregates.items()],
        output_dir / "summary.csv",
    )

    markdown = build_markdown_report(metadata, classification=payload)
    (output_dir / "evaluation_report.md").write_text(markdown, encoding="utf-8")

    logger.info("wrote evaluation artifacts to %s", output_dir)
    print(f"\npositive_macro_f1 : {report.aggregates['positive_macro_f1']:.4f}")
    print(f"macro_auroc       : {report.aggregates['macro_auroc']:.4f}")
    print(f"macro_auprc       : {report.aggregates['macro_auprc']:.4f}")
    print(f"report            : {output_dir / 'evaluation_report.md'}")
    return 0


def _write_plots(predictions, report, args, output_dir: Path) -> None:
    from training.evaluation import visualization

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    y_true, valid = binarize_labels(predictions.labels, args.uncertain_policy)
    scores = predictions.positive_probabilities
    names = predictions.pathology_names

    visualization.plot_roc_curves(scores, y_true, valid, names, plots_dir)
    visualization.plot_pr_curves(scores, y_true, valid, names, plots_dir)
    visualization.plot_probability_histogram(scores, y_true, valid, names, plots_dir)
    visualization.plot_reliability_diagram(scores, y_true, valid, plots_dir)
    visualization.plot_confusion_matrices(
        report.three_class_confusion, names, CLASS_NAMES, output_dir
    )
    visualization.plot_per_pathology_bars(
        {m.name: m.f1 for m in report.per_pathology},
        "Positive-class F1 by pathology",
        "F1",
        plots_dir,
        "positive_f1_by_pathology.png",
    )
    visualization.plot_per_pathology_bars(
        {m.name: m.prevalence for m in report.per_pathology},
        "Positive prevalence by pathology",
        "Prevalence",
        plots_dir,
        "prevalence_by_pathology.png",
    )
    logger.info("wrote plots to %s", plots_dir)


if __name__ == "__main__":
    raise SystemExit(main())
