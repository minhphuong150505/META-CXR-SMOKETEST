#!/usr/bin/env python3
"""Calibrate per-pathology positive-class thresholds on the validation split.

    python scripts/calibrate_thresholds.py \\
        --predictions outputs/val_predictions.npz \\
        --objective f1 \\
        --uncertain-policy ignore_uncertain \\
        --output outputs/thresholds.json

The split name is written into the output file, and ``evaluate_stage1.py``
refuses a threshold file recorded against a test split. Calibrating a threshold
on test and then reporting test metrics with it is the classic way to publish an
optimistically biased number; the two scripts are wired so it cannot happen by
accident.

Exit codes: 0 success, 1 calibration failed, 2 bad arguments or input.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.evaluation.schemas import ClassificationPredictions  # noqa: E402
from training.evaluation.threshold_calibration import (  # noqa: E402
    DEFAULT_OBJECTIVE,
    OBJECTIVES,
    CalibrationError,
    calibrate_thresholds,
)
from training.evaluation.uncertain_policy import DEFAULT_POLICY, POLICIES  # noqa: E402

logger = logging.getLogger("calibrate_thresholds")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--objective", default=DEFAULT_OBJECTIVE, choices=sorted(OBJECTIVES))
    parser.add_argument("--uncertain-policy", default=DEFAULT_POLICY, choices=sorted(POLICIES))
    parser.add_argument(
        "--constraint",
        type=float,
        default=0.5,
        help="Precision floor for recall_at_precision, or recall floor for "
        "precision_at_recall. Ignored by the other objectives.",
    )
    parser.add_argument(
        "--split",
        default="validation",
        help="Split these predictions came from. A test split is refused.",
    )
    parser.add_argument(
        "--min-positive",
        type=int,
        default=1,
        help="Pathologies with fewer positives keep the 0.5 default.",
    )
    parser.add_argument("--output", required=True, type=Path)
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
    except Exception as exc:  # noqa: BLE001
        logger.error("could not read %s: %s", args.predictions, exc)
        return 2

    if predictions.num_samples == 0:
        logger.error("%s contains no samples", args.predictions)
        return 2

    logger.info(
        "calibrating %d pathologies on %d studies from split %r",
        predictions.num_pathologies,
        predictions.num_samples,
        args.split,
    )

    try:
        result = calibrate_thresholds(
            predictions,
            split=args.split,
            objective=args.objective,
            uncertain_policy=args.uncertain_policy,
            constraint=args.constraint,
            min_positive=args.min_positive,
        )
    except CalibrationError as exc:
        logger.error("%s", exc)
        return 2

    result.save(args.output)

    calibrated = [t for t in result.thresholds if t.calibrated]
    skipped = [t for t in result.thresholds if not t.calibrated]
    logger.info("calibrated %d pathologies, %d kept the default", len(calibrated), len(skipped))
    for threshold in skipped:
        logger.warning("%s kept threshold 0.5: %s", threshold.name, threshold.reason)

    print(f"\n{'pathology':<30} {'thr':>6} {'obj@thr':>9} {'obj@0.5':>9} {'n+':>6}")
    print("-" * 64)
    for threshold in result.thresholds:
        objective_score = (
            "n/a" if threshold.objective_score != threshold.objective_score
            else f"{threshold.objective_score:.4f}"
        )
        default_score = (
            "n/a" if threshold.default_threshold_score != threshold.default_threshold_score
            else f"{threshold.default_threshold_score:.4f}"
        )
        print(
            f"{threshold.name:<30} {threshold.threshold:>6.3f} {objective_score:>9} "
            f"{default_score:>9} {threshold.n_positive:>6}"
        )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
