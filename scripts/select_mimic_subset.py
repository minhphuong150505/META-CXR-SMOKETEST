#!/usr/bin/env python3
"""Select a deterministic, whole-subject MIMIC-CXR subset under a byte cap.

The input GCS inventory is metadata-only output from
``gcloud storage ls --json 'gs://BUCKET/files/**'``. No image is downloaded.
Validation and test subjects are retained in full; the train remainder is
ranked by inverse positive-label frequency with a seeded hash tie-breaker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


IMAGE_RE = re.compile(
    r"^files/p(?P<prefix>1\d)/p(?P<subject_id>\d+)/s(?P<study_id>\d+)/"
    r"(?P<dicom_id>[0-9a-f-]+)\.jpg$"
)
KEYS = ["subject_id", "study_id", "dicom_id"]
SUBJECT_KEY = ["subject_id"]
GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-json", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--budget-bytes", type=int, default=200 * GIB)
    parser.add_argument("--artifact-reserve-bytes", type=int, default=2 * GIB)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_inventory(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for item in payload:
        metadata = item.get("metadata", {})
        name = str(metadata.get("name", ""))
        match = IMAGE_RE.fullmatch(name)
        if not match:
            raise ValueError(f"Non-canonical image object in inventory: {name!r}")
        row = match.groupdict()
        row.update(
            {
                "object_path": name,
                "object_size": int(metadata["size"]),
                "crc32c": metadata.get("crc32c"),
                "md5_hash": metadata.get("md5Hash"),
                "generation": metadata.get("generation"),
            }
        )
        records.append(row)
    frame = pd.DataFrame.from_records(records)
    for column in ("subject_id", "study_id"):
        frame[column] = frame[column].astype("int64")
    if frame["dicom_id"].duplicated().any() or frame["object_path"].duplicated().any():
        raise ValueError("Inventory contains duplicate DICOM IDs or object paths")
    return frame.sort_values("object_path", kind="stable").reset_index(drop=True)


def _seeded_tie(subject_id: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{subject_id}".encode()).hexdigest()


def choose_subjects(
    rows: pd.DataFrame,
    chexpert: pd.DataFrame,
    image_budget: int,
    seed: int,
) -> set[int]:
    subject_summary = rows.groupby("subject_id", sort=True).agg(
        split=("split", "first"),
        split_count=("split", "nunique"),
        object_size=("object_size", "sum"),
        images=("dicom_id", "size"),
    )
    if (subject_summary["split_count"] != 1).any():
        raise ValueError("A subject appears in more than one original split")

    fixed = subject_summary[subject_summary["split"].isin(["validate", "test"])]
    fixed_bytes = int(fixed["object_size"].sum())
    if fixed_bytes > image_budget:
        raise ValueError("Full original validation+test subjects exceed the image budget")

    label_cols = [c for c in chexpert.columns if c not in ("subject_id", "study_id")]
    train_labels = chexpert.merge(
        rows.loc[rows["split"] == "train", ["subject_id", "study_id"]].drop_duplicates(),
        on=["subject_id", "study_id"],
        how="inner",
        validate="one_to_one",
    )
    positive_by_subject = (
        train_labels.assign(**{c: train_labels[c].eq(1.0) for c in label_cols})
        .groupby("subject_id", sort=True)[label_cols]
        .max()
    )
    positive_support = positive_by_subject.sum(axis=0).clip(lower=1)
    rarity = positive_by_subject.mul(1.0 / positive_support, axis=1).sum(axis=1)

    train = subject_summary[subject_summary["split"] == "train"].copy()
    train["rarity_score"] = rarity.reindex(train.index).fillna(0.0)
    train["positive_labels"] = positive_by_subject.sum(axis=1).reindex(train.index).fillna(0)
    train["tie"] = [_seeded_tie(int(subject_id), seed) for subject_id in train.index]
    train = train.sort_values(
        ["rarity_score", "positive_labels", "tie"],
        ascending=[False, False, True],
        kind="stable",
    )

    selected = set(int(value) for value in fixed.index)
    used = fixed_bytes
    for subject_id, row in train.iterrows():
        size = int(row["object_size"])
        if used + size <= image_budget:
            selected.add(int(subject_id))
            used += size
    if not any(subject_id in selected for subject_id in train.index):
        raise ValueError("Budget left no train subject")
    return selected


def main() -> None:
    args = parse_args()
    if args.budget_bytes <= 0 or args.artifact_reserve_bytes <= 0:
        raise ValueError("Budget and artifact reserve must be positive")
    image_budget = args.budget_bytes - args.artifact_reserve_bytes
    if image_budget <= 0:
        raise ValueError("Artifact reserve consumes the entire budget")

    raw_dir = Path(args.raw_dir)
    output_root = Path(args.output_root)
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    inventory = load_inventory(Path(args.inventory_json))
    metadata = pd.read_csv(raw_dir / "mimic-cxr-2.0.0-metadata.csv.gz")
    split = pd.read_csv(raw_dir / "mimic-cxr-2.0.0-split.csv.gz")
    chexpert = pd.read_csv(raw_dir / "mimic-cxr-2.0.0-chexpert.csv.gz")
    for frame in (metadata, split, chexpert):
        frame["subject_id"] = frame["subject_id"].astype("int64")
        frame["study_id"] = frame["study_id"].astype("int64")
    metadata["dicom_id"] = metadata["dicom_id"].astype(str)
    split["dicom_id"] = split["dicom_id"].astype(str)

    rows = inventory.merge(
        metadata[[*KEYS, "ViewPosition"]], on=KEYS, how="inner", validate="one_to_one"
    ).merge(split[[*KEYS, "split"]], on=KEYS, how="inner", validate="one_to_one")
    if len(rows) != len(inventory) or len(rows) != len(metadata):
        raise ValueError(
            "GCS inventory, metadata and split rows do not form an exact one-to-one set"
        )
    if rows.groupby("subject_id")["split"].nunique().max() != 1:
        raise ValueError("Original split leaks at subject level")

    selected_subjects = choose_subjects(rows, chexpert, image_budget, args.seed)
    selected = rows[rows["subject_id"].isin(selected_subjects)].copy()
    selected = selected.sort_values("object_path", kind="stable").reset_index(drop=True)

    selected_keys = selected[KEYS]
    metadata_subset = metadata.merge(selected_keys, on=KEYS, how="inner", validate="one_to_one")
    split_subset = split.merge(selected_keys, on=KEYS, how="inner", validate="one_to_one")
    selected_studies = selected[["subject_id", "study_id"]].drop_duplicates()
    chexpert_subset = chexpert.merge(
        selected_studies,
        on=["subject_id", "study_id"],
        how="inner",
        validate="one_to_one",
    )

    raw_outputs = {
        "metadata": output_root / "mimic-cxr-2.0.0-metadata-subset.csv.gz",
        "split": output_root / "mimic-cxr-2.0.0-split-subset.csv.gz",
        "chexpert": output_root / "mimic-cxr-2.0.0-chexpert-subset.csv.gz",
    }
    metadata_subset.to_csv(raw_outputs["metadata"], index=False, compression="gzip")
    split_subset.to_csv(raw_outputs["split"], index=False, compression="gzip")
    chexpert_subset.to_csv(raw_outputs["chexpert"], index=False, compression="gzip")

    # Production preprocessing expects canonical input names in its raw dir.
    canonical_raw = output_root / "raw_for_preprocess"
    canonical_raw.mkdir(exist_ok=True)
    metadata_subset.to_csv(
        canonical_raw / "mimic-cxr-2.0.0-metadata.csv.gz", index=False, compression="gzip"
    )
    split_subset.to_csv(
        canonical_raw / "mimic-cxr-2.0.0-split.csv.gz", index=False, compression="gzip"
    )
    chexpert_subset.to_csv(
        canonical_raw / "mimic-cxr-2.0.0-chexpert.csv.gz", index=False, compression="gzip"
    )

    selected.to_parquet(manifests_dir / "object_inventory.parquet", index=False)
    selected[[
        "subject_id", "study_id", "dicom_id", "prefix", "split", "ViewPosition",
        "object_path", "object_size", "crc32c", "md5_hash", "generation",
    ]].to_parquet(manifests_dir / "selection_manifest.parquet", index=False)
    transfer = selected[["object_path", "generation"]].copy()
    transfer.to_csv(manifests_dir / "gcs_transfer_manifest.csv", index=False, header=False)

    actual_known = int(selected["object_size"].sum()) + sum(
        path.stat().st_size for path in raw_outputs.values()
    )
    if actual_known > args.budget_bytes:
        raise ValueError("Selected images and subset metadata exceed the total byte budget")

    label_cols = [c for c in chexpert.columns if c not in ("subject_id", "study_id")]
    selected_label_rows = chexpert_subset.merge(
        split_subset[["subject_id", "study_id", "split"]].drop_duplicates(),
        on=["subject_id", "study_id"], how="left", validate="one_to_one"
    )
    support = {}
    for split_name, frame in selected_label_rows.groupby("split"):
        support[split_name] = {
            label: {
                "positive": int(frame[label].eq(1.0).sum()),
                "negative": int(frame[label].eq(0.0).sum()),
                "uncertain": int(frame[label].eq(-1.0).sum()),
                "missing": int(frame[label].isna().sum()),
            }
            for label in label_cols
        }
    summary = {
        "algorithm": "whole_subject_fixed_val_test_inverse_positive_frequency_v1",
        "seed": args.seed,
        "budget_bytes": args.budget_bytes,
        "artifact_reserve_bytes": args.artifact_reserve_bytes,
        "selected_image_bytes": int(selected["object_size"].sum()),
        "known_bytes_before_processed_artifacts": actual_known,
        "subjects": int(selected["subject_id"].nunique()),
        "studies": int(selected[["subject_id", "study_id"]].drop_duplicates().shape[0]),
        "images": int(len(selected)),
        "split_images": {k: int(v) for k, v in selected["split"].value_counts().items()},
        "prefix_images": {k: int(v) for k, v in selected["prefix"].value_counts().items()},
        "label_support": support,
    }
    (manifests_dir / "selection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "label_support"}, indent=2))


if __name__ == "__main__":
    main()
