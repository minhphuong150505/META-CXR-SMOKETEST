#!/usr/bin/env python3
"""Fail-closed QA and manifest generation for the private MIMIC subset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image


PATHOLOGIES = (
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices",
)
UPLOAD_EXCLUDES = {"raw_for_preprocess"}
GENERATED_ARTIFACTS = {
    "dataset_manifest.json",
    "manifests/qa_report.json",
    "manifests/qa_summary.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--source-bucket", required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--uploaded-inventory-json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode-per-split", type=int, default=3)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gcloud_inventory(path: Path) -> pd.DataFrame:
    rows = []
    for item in json.loads(path.read_text(encoding="utf-8")):
        metadata = item.get("metadata", {})
        rows.append(
            {
                "object_path": metadata.get("name"),
                "object_size": int(metadata.get("size", 0)),
                "crc32c": metadata.get("crc32c"),
            }
        )
    return pd.DataFrame(rows)


def decode_samples(frame: pd.DataFrame, bucket: str, count: int, seed: int) -> int:
    decoded = 0
    for offset, (split_name, split_frame) in enumerate(frame.groupby("split", sort=True)):
        sample = split_frame.sample(n=min(count, len(split_frame)), random_state=seed + offset)
        for object_path in sample["object_path"]:
            result = subprocess.run(
                ["gcloud", "storage", "cat", f"gs://{bucket}/{object_path}"],
                check=True,
                capture_output=True,
            )
            with Image.open(io.BytesIO(result.stdout)) as image:
                image.verify()
            decoded += 1
    return decoded


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root)
    manifests = root / "manifests"
    selection = pd.read_parquet(manifests / "selection_manifest.parquet")
    if selection["dicom_id"].duplicated().any():
        raise ValueError("duplicate dicom_id in selection manifest")
    if selection["object_path"].duplicated().any():
        raise ValueError("duplicate object path in selection manifest")

    processed_frames = []
    for name, source_split in (("train", "train"), ("val", "validate"), ("test", "test")):
        csv_path = root / "processed" / f"{name}.csv"
        parquet_path = root / "processed" / f"{name}.parquet"
        if not csv_path.is_file() or not parquet_path.is_file():
            raise FileNotFoundError(f"Missing processed pair for split {name}")
        csv_frame = pd.read_csv(csv_path)
        parquet_frame = pd.read_parquet(parquet_path)
        if len(csv_frame) != len(parquet_frame):
            raise ValueError(f"CSV/parquet row count mismatch for {name}")
        if csv_frame.empty:
            raise ValueError(f"Processed split {name} is empty")
        csv_frame["expected_split"] = source_split
        processed_frames.append(csv_frame)
    processed = pd.concat(processed_frames, ignore_index=True)

    required = {
        "subject_id", "study_id", "dicom_id", "split", "ViewPosition", "image_path",
        "findings_clean", "impression_clean", "extraction_method", "target_valid",
        "impression_valid", "has_chexpert_label",
    }
    missing = sorted(required - set(processed.columns))
    if missing:
        raise ValueError(f"Processed schema missing columns: {missing}")
    if processed["dicom_id"].duplicated().any():
        raise ValueError("duplicate dicom_id in processed splits")
    for column in ("target_valid", "impression_valid", "has_chexpert_label"):
        values = processed[column].astype(str).str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError(f"Invalid boolean values in {column}")
    if not (processed["split"] == processed["expected_split"]).all():
        raise ValueError("Processed rows do not preserve the original split")
    if (processed.groupby(["subject_id", "study_id"])["findings_clean"].nunique() > 1).any():
        raise ValueError("FINDINGS differs across views in one study")
    if processed.groupby("subject_id")["split"].nunique().max() != 1:
        raise ValueError("subject leakage across splits")
    if processed.groupby(["subject_id", "study_id"])["split"].nunique().max() != 1:
        raise ValueError("study leakage across splits")

    selected_keys = set(selection["dicom_id"].astype(str))
    processed_keys = set(processed["dicom_id"].astype(str))
    if selected_keys != processed_keys:
        raise ValueError("Processed image rows do not exactly match selection manifest")
    expected_paths = set(selection["object_path"].astype(str))
    if set(processed["image_path"].astype(str)) != expected_paths:
        raise ValueError("Processed relative image paths do not match selected objects")

    split = pd.read_csv(root / "mimic-cxr-2.0.0-split-subset.csv.gz")
    chexpert = pd.read_csv(root / "mimic-cxr-2.0.0-chexpert-subset.csv.gz")
    if chexpert.duplicated(["subject_id", "study_id"]).any():
        raise ValueError("CheXpert subset is not many-to-one by study")
    studies = split[["subject_id", "study_id", "split"]].drop_duplicates()
    if studies.duplicated(["subject_id", "study_id"]).any():
        raise ValueError("Study belongs to more than one split")
    labels = chexpert.merge(
        studies, on=["subject_id", "study_id"], how="inner", validate="one_to_one"
    )
    support = {}
    for split_name, frame in labels.groupby("split"):
        support[split_name] = {}
        for pathology in PATHOLOGIES:
            counts = {
                "positive": int(frame[pathology].eq(1.0).sum()),
                "negative": int(frame[pathology].eq(0.0).sum()),
                "uncertain": int(frame[pathology].eq(-1.0).sum()),
                "missing": int(frame[pathology].isna().sum()),
            }
            support[split_name][pathology] = counts
            if split_name in {"validate", "test"} and counts["positive"] == 0:
                raise ValueError(
                    f"{pathology} has zero positive support in {split_name}"
                )

    decoded = decode_samples(selection, args.source_bucket, args.decode_per_split, args.seed)

    if args.uploaded_inventory_json:
        uploaded = load_gcloud_inventory(Path(args.uploaded_inventory_json))
        expected = selection[["object_path", "object_size", "crc32c"]].copy()
        comparison = expected.merge(
            uploaded, on="object_path", how="outer", suffixes=("_expected", "_uploaded"),
            indicator=True,
        )
        if not (comparison["_merge"] == "both").all():
            raise ValueError("Uploaded object set differs from selection")
        if not (
            comparison["object_size_expected"] == comparison["object_size_uploaded"]
        ).all():
            raise ValueError("Uploaded object sizes differ from source inventory")
        known_crc = comparison["crc32c_expected"].notna()
        if not (
            comparison.loc[known_crc, "crc32c_expected"]
            == comparison.loc[known_crc, "crc32c_uploaded"]
        ).all():
            raise ValueError("Uploaded CRC32C differs from source inventory")

    artifact_files = []
    base_artifact_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in UPLOAD_EXCLUDES for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix()
        if relative in GENERATED_ARTIFACTS:
            continue
        size = path.stat().st_size
        base_artifact_bytes += size
        artifact_files.append(
            {"path": relative, "bytes": size, "sha256": sha256_file(path)}
        )
    image_bytes = int(selection["object_size"].sum())
    qa_path = manifests / "qa_report.json"
    manifest_path = root / "dataset_manifest.json"
    summary_path = manifests / "qa_summary.md"
    generated_bytes = 0
    for _ in range(10):
        artifact_bytes = base_artifact_bytes + generated_bytes
        total_bytes = image_bytes + artifact_bytes
        summary = {
            "status": "pass",
            "seed": args.seed,
            "budget_bytes": args.budget_bytes,
            "actual_bytes": total_bytes,
            "actual_gib": f"{total_bytes / 1024**3:.6f}",
            "image_bytes": image_bytes,
            "artifact_bytes": artifact_bytes,
            "generated_artifact_bytes": generated_bytes,
            "subjects": int(selection["subject_id"].nunique()),
            "studies": int(selection[["subject_id", "study_id"]].drop_duplicates().shape[0]),
            "images": int(len(selection)),
            "split_images": {k: int(v) for k, v in selection["split"].value_counts().items()},
            "split_studies": {
                k: int(v)
                for k, v in selection.drop_duplicates(["subject_id", "study_id"])["split"]
                .value_counts()
                .items()
            },
            "view_counts": {
                str(k): int(v)
                for k, v in selection["ViewPosition"].fillna("UNKNOWN").value_counts().items()
            },
            "decoded_samples": decoded,
            "label_support": support,
            "artifact_files": artifact_files,
        }
        qa_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "status": "qa_passed",
            "selection_policy": "whole_subject_fixed_val_test_inverse_positive_frequency_v1",
            "seed": args.seed,
            "budget_bytes": args.budget_bytes,
            "actual_bytes": total_bytes,
            "counts": {
                "subjects": summary["subjects"],
                "studies": summary["studies"],
                "images": summary["images"],
                "split_images": summary["split_images"],
                "split_studies": summary["split_studies"],
            },
            "object_inventory_sha256": sha256_file(manifests / "object_inventory.parquet"),
            "selection_manifest_sha256": sha256_file(manifests / "selection_manifest.parquet"),
            "qa_report_sha256": sha256_file(qa_path),
            "required_paths": [
                "files/", "processed/train.csv", "processed/val.csv", "processed/test.csv",
                "mimic-cxr-2.0.0-chexpert-subset.csv.gz",
            ],
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        lines = [
            "# Dataset QA", "", "| Check | Value |", "|---|---:|",
            f"| Status | {summary['status']} |",
            f"| Bytes | {total_bytes} |", f"| GiB | {summary['actual_gib']} |",
            f"| Subjects | {summary['subjects']} |", f"| Studies | {summary['studies']} |",
            f"| Images | {summary['images']} |", f"| Decoded samples | {decoded} |",
        ]
        summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        measured_generated_bytes = sum(
            path.stat().st_size for path in (manifest_path, qa_path, summary_path)
        )
        if measured_generated_bytes == generated_bytes:
            break
        generated_bytes = measured_generated_bytes
    else:
        raise RuntimeError("Generated manifest sizes did not converge")

    if total_bytes > args.budget_bytes:
        raise ValueError(f"Actual dataset bytes {total_bytes} exceed budget {args.budget_bytes}")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"label_support", "artifact_files"}}, indent=2))


if __name__ == "__main__":
    main()
