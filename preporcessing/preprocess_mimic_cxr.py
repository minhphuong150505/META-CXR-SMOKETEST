#!/usr/bin/env python3
"""Preprocess the full MIMIC-CXR (p10-p19) metadata + reports into train/val/test CSVs.

Script form of `kltn-data-preprocessing.ipynb`, with the Kaggle/gdown plumbing and the
exploratory plotting cells removed. Reads the three `.csv.gz` metadata files and the
per-study report .txt tree, emits the split CSVs consumed by
`model/lavis/data/ReportDataset.py::MIMIC_CXR_Dataset`.

Only CSVs and reports are touched — images are never read.

Important differences from the superseded notebook:
  * FINDINGS and IMPRESSION are parsed at study level (~227k rows) and merged onto the
    image level afterwards, each with its own validity flag (``target_valid`` /
    ``impression_valid``). Neither section is ever substituted for the other, so a
    ``findings_and_impression`` target must check both flags. Studies without usable
    FINDINGS remain available with ``target_valid=false``.
  * Target-length bounds are derived only from train-study lexical-token counts, and
    FINDINGS and IMPRESSION get separate bounds because their length distributions
    differ; invalid targets are blanked so a teacher branch cannot consume them.
  * `image_path` is written RELATIVE (`files/p1X/pXXXXXXXX/sYYYYYYY/<dicom>.jpg`) because
    `ReportDataset._row_visual` re-anchors it with `os.path.join(vis_root, rel)`.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from .mimic_report_parser import (
        clean_report_text,
        count_lexical_tokens,
        extract_sections,
        get_target_text,
    )
except ImportError:  # Direct execution: python preporcessing/preprocess_mimic_cxr.py
    from mimic_report_parser import (  # type: ignore
        clean_report_text,
        count_lexical_tokens,
        extract_sections,
        get_target_text,
    )

# split.csv uses "validate"; the pipeline expects val.csv
SPLIT_TO_FILENAME = {"train": "train", "validate": "val", "test": "test"}
FRONTAL_VIEWS = ["PA", "AP"]
MIN_IMAGE_SIZE = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", required=True,
                   help="dir holding mimic-cxr-2.0.0-{chexpert,metadata,split}.csv.gz")
    p.add_argument("--reports-root", required=True,
                   help="root of the report tree, i.e. the dir containing p10/ .. p19/")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--views", choices=["all", "frontal"], default="all",
                   help="'all' keeps every ViewPosition (NaN -> UNKNOWN); "
                        "'frontal' keeps only PA/AP")
    p.add_argument("--workers", type=int, default=16, help="threads for reading reports")
    p.add_argument("--min-tokens", "--min-words", dest="min_tokens", type=int, default=3,
                   help="mask generative targets with fewer lexical tokens than this "
                        "(--min-words is retained as a deprecated alias)")
    p.add_argument("--upper-quantile", type=float, default=0.99,
                   help="drop rows above this quantile of findings length")
    p.add_argument("--limit-studies", type=int, default=None,
                   help="debug: only process this many studies")
    return p.parse_args()


def clean_chexpert(chexpert_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Notebook cell 12. Keeps label-less studies, flags them via has_chexpert_label."""
    chexpert_df = chexpert_df.copy()
    chexpert_df["subject_id"] = chexpert_df["subject_id"].astype(int)
    chexpert_df["study_id"] = chexpert_df["study_id"].astype(int)

    key_cols = ["subject_id", "study_id"]
    dup_count = int(chexpert_df.duplicated(key_cols).sum())
    print(f"[chexpert] duplicated (subject_id, study_id): {dup_count}")
    if dup_count:
        raise ValueError(
            "CheXpert rows are not unique by (subject_id, study_id); refusing a "
            "many-to-many label merge."
        )

    label_cols = [c for c in chexpert_df.columns if c not in ("subject_id", "study_id")]
    print(f"[chexpert] label columns: {len(label_cols)}")

    no_label_mask = chexpert_df[label_cols].isnull().all(axis=1)
    print(f"[chexpert] studies with no label at all (kept, flagged): {int(no_label_mask.sum())}")

    out = chexpert_df.copy()
    out["has_chexpert_label"] = ~no_label_mask

    # 3-class mapping: 0=negative, 1=positive, 2=uncertain, NaN->0.
    # For has_chexpert_label=False rows the 0s are placeholders, NOT real negatives.
    for col in label_cols:
        mapped = pd.Series(0, index=out.index, dtype="int8")
        mapped[out[col] == 1.0] = 1
        mapped[out[col] == -1.0] = 2
        out[col] = mapped

    return out, label_cols


def clean_metadata(metadata_df: pd.DataFrame, views: str) -> pd.DataFrame:
    """Notebook cell 14, minus the boxplots."""
    metadata_df = metadata_df.copy()
    for col in ("subject_id", "study_id"):
        if col in metadata_df.columns:
            metadata_df[col] = metadata_df[col].astype(int)
    metadata_df["dicom_id"] = metadata_df["dicom_id"].astype(str)

    before = metadata_df.shape
    metadata_df = metadata_df.drop_duplicates().reset_index(drop=True)
    print(f"[metadata] {before} -> {metadata_df.shape} after dropping fully-identical rows")

    key_cols = ["dicom_id", "subject_id", "study_id"]
    if metadata_df.duplicated(key_cols).any():
        raise ValueError(
            "Metadata is not unique by (dicom_id, subject_id, study_id); refusing "
            "a many-to-many split merge."
        )

    if metadata_df["ViewPosition"].dtype == object:
        metadata_df["ViewPosition"] = metadata_df["ViewPosition"].str.strip()
    metadata_df["ViewPosition"] = metadata_df["ViewPosition"].replace("", np.nan)

    # Drop only implausibly small images; high-res is normal for CXR.
    too_small = (metadata_df["Rows"] < MIN_IMAGE_SIZE) | (metadata_df["Columns"] < MIN_IMAGE_SIZE)
    print(f"[metadata] images smaller than {MIN_IMAGE_SIZE}px: {int(too_small.sum())}")
    metadata_df = metadata_df[~too_small].reset_index(drop=True)

    if views == "frontal":
        metadata_df = metadata_df[metadata_df["ViewPosition"].isin(FRONTAL_VIEWS)].reset_index(drop=True)
        print(f"[metadata] frontal-only filter -> {metadata_df.shape}")
    else:
        metadata_df["ViewPosition"] = metadata_df["ViewPosition"].fillna("UNKNOWN")

    print("[metadata] ViewPosition distribution:")
    print(metadata_df["ViewPosition"].value_counts().to_string())
    return metadata_df


def clean_split(split_df: pd.DataFrame) -> pd.DataFrame:
    """Notebook cell 19."""
    split_df = split_df.copy()
    split_df["subject_id"] = split_df["subject_id"].astype(int)
    split_df["study_id"] = split_df["study_id"].astype(int)
    split_df["dicom_id"] = split_df["dicom_id"].astype(str)

    before = split_df.shape
    split_df = split_df.drop_duplicates().reset_index(drop=True)
    print(f"[split] {before} -> {split_df.shape} after dropping duplicates")

    split_df["split"] = split_df["split"].str.strip().str.lower()

    expected = set(SPLIT_TO_FILENAME)
    unexpected = sorted(set(split_df["split"].dropna()) - expected)
    if unexpected:
        raise ValueError(f"Unexpected split values: {unexpected}")

    key_cols = ["dicom_id", "subject_id", "study_id"]
    if split_df.duplicated(key_cols).any():
        raise ValueError(
            "Split rows are not unique by (dicom_id, subject_id, study_id)."
        )

    subject_leakage = int((split_df.groupby("subject_id")["split"].nunique() > 1).sum())
    study_leakage = int((split_df.groupby(["subject_id", "study_id"])["split"].nunique() > 1).sum())
    print(f"[split] subjects appearing in more than one split (expect 0): {subject_leakage}")
    print(f"[split] studies appearing in more than one split (expect 0): {study_leakage}")
    if subject_leakage or study_leakage:
        raise ValueError(
            "Patient/study leakage detected across train/validate/test; refusing "
            "to produce training files."
        )

    print("[split] distribution:")
    print(split_df["split"].value_counts().to_string())
    return split_df


def report_path(reports_root: Path, subject_id: int, study_id: int) -> Path:
    sid = str(subject_id)
    return reports_root / f"p{sid[:2]}" / f"p{sid}" / f"s{study_id}.txt"


def build_study_text(studies: pd.DataFrame, reports_root: Path, workers: int) -> pd.DataFrame:
    """Read + parse every report once, at study level (notebook cells 29/32/34 fused).

    Returns one row per (subject_id, study_id) with the cleaned text columns only —
    report_raw is dropped before returning so it never reaches the image-level frame.
    """
    records = studies.to_dict("records")
    print(f"[reports] reading + parsing {len(records)} studies with {workers} threads")

    def read_and_parse(rec: dict) -> tuple[str, str, str] | None:
        """Read, section and clean one study, returning only short strings.

        Parsing inside the worker lets each report's raw text be freed as soon
        as it is consumed. Materialising all 227,835 raw reports (~1.2 GB) and
        their parse output at the same time exhausted memory on a 16 GB box.
        """
        path = report_path(reports_root, rec["subject_id"], rec["study_id"])
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, UnicodeDecodeError):
            return None
        findings, impression, method = get_target_text(text)
        return clean_report_text(findings), clean_report_text(impression), method

    with ThreadPoolExecutor(max_workers=workers) as ex:
        parsed = list(
            tqdm(ex.map(read_and_parse, records), total=len(records), desc="reports")
        )

    n_missing = sum(p is None for p in parsed)
    fail_rate = n_missing / len(parsed) * 100 if parsed else 0.0
    print(f"[reports] read ok: {len(parsed) - n_missing}, missing: {n_missing} ({fail_rate:.2f}%)")
    if fail_rate > 5:
        raise SystemExit(
            f"[reports] {fail_rate:.2f}% of reports unreadable — check --reports-root "
            f"structure before trusting this output"
        )

    out = pd.DataFrame(records)
    # IMPRESSION is parsed from an explicit IMPRESSION/CONCLUSION tag only. It is
    # never recovered from the narrative body, so an empty value here means the
    # report genuinely had no impression section rather than a parse miss.
    out["findings_clean"] = [p[0] if p else "" for p in parsed]
    out["impression_clean"] = [p[1] if p else "" for p in parsed]
    out["extraction_method"] = [p[2] if p else "MISSING_REPORT" for p in parsed]
    del parsed
    out["target_valid"] = out["findings_clean"].str.len().gt(0)
    out["impression_valid"] = out["impression_clean"].str.len().gt(0)

    print("[reports] extraction method distribution:")
    print(out["extraction_method"].value_counts().to_string())
    print(f"[reports] impression present: {int(out['impression_valid'].sum())}/{len(out)} "
          f"({out['impression_valid'].mean() * 100:.2f}%)")
    print(f"[reports] findings+impression both present: "
          f"{int((out['target_valid'] & out['impression_valid']).sum())}")
    return out


def build_image_path(subject_id: int, study_id: int, dicom_id: str) -> str:
    """Relative path; ReportDataset joins it onto vis_root."""
    sid = str(subject_id)
    return f"files/p{sid[:2]}/p{sid}/s{study_id}/{dicom_id}.jpg"


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.min_tokens < 0:
        raise ValueError("--min-tokens must be non-negative")
    if not 0 < args.upper_quantile <= 1:
        raise ValueError("--upper-quantile must be in (0, 1]")
    if args.limit_studies is not None and args.limit_studies < 1:
        raise ValueError("--limit-studies must be positive")
    raw_dir = Path(args.raw_dir)
    reports_root = Path(args.reports_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== load ===")
    chexpert_df = pd.read_csv(raw_dir / "mimic-cxr-2.0.0-chexpert.csv.gz")
    split_df = pd.read_csv(raw_dir / "mimic-cxr-2.0.0-split.csv.gz")
    metadata_df = pd.read_csv(raw_dir / "mimic-cxr-2.0.0-metadata.csv.gz")
    print(f"chexpert={chexpert_df.shape} split={split_df.shape} metadata={metadata_df.shape}")

    print("\n=== clean chexpert ===")
    chexpert_clean, label_cols = clean_chexpert(chexpert_df)
    del chexpert_df

    print("\n=== clean metadata ===")
    metadata_clean = clean_metadata(metadata_df, args.views)
    del metadata_df

    print("\n=== clean split ===")
    split_clean = clean_split(split_df)
    del split_df

    print("\n=== merge ===")
    # metadata carries many DICOM columns we do not need downstream; keep it narrow
    meta_cols = ["dicom_id", "subject_id", "study_id", "ViewPosition"]
    merged = split_clean.merge(
        metadata_clean[meta_cols],
        on=["dicom_id", "subject_id", "study_id"],
        how="inner",
        validate="one_to_one",
    )
    print(f"after split+metadata: {merged.shape}")
    merged = merged.merge(
        chexpert_clean[["subject_id", "study_id", "has_chexpert_label"]],
        on=["subject_id", "study_id"],
        how="left",
        validate="many_to_one",
    )
    missing_chexpert = int(merged["has_chexpert_label"].isna().sum())
    if missing_chexpert:
        print(f"[chexpert] {missing_chexpert} image rows have no CheXpert record; "
              "kept with classification masked")
    merged["has_chexpert_label"] = merged["has_chexpert_label"].fillna(False).astype(bool)
    print(f"after chexpert: {merged.shape}")
    print(f"unique studies: {merged['study_id'].nunique()} (paper reports 227,835)")
    del metadata_clean, split_clean, chexpert_clean

    print("\n=== reports ===")
    studies = merged[["subject_id", "study_id"]].drop_duplicates().reset_index(drop=True)
    if args.limit_studies:
        studies = studies.head(args.limit_studies)
        print(f"[debug] limited to {len(studies)} studies")
    study_text = build_study_text(studies, reports_root, args.workers)

    merged = merged.merge(
        study_text[["subject_id", "study_id", "findings_clean", "impression_clean",
                    "extraction_method", "target_valid", "impression_valid"]],
        on=["subject_id", "study_id"],
        how="inner",
        validate="many_to_one",
    )
    print(f"after attaching report text: {merged.shape}")
    del study_text

    print("\n=== target length mask ===")
    merged["findings_token_count"] = merged["findings_clean"].map(count_lexical_tokens).astype(int)
    # Retain the old diagnostic column for downstream notebooks, but do not use
    # its whitespace count for filtering.
    merged["findings_word_count"] = merged["findings_clean"].str.split().str.len().fillna(0).astype(int)
    study_targets = merged.drop_duplicates(["subject_id", "study_id"])
    train_lengths = study_targets.loc[
        (study_targets["split"] == "train") & study_targets["target_valid"],
        "findings_token_count",
    ]
    if train_lengths.empty:
        raise ValueError("No valid train FINDINGS targets; check report parsing and split inputs.")
    upper = int(np.ceil(train_lengths.quantile(args.upper_quantile)))
    too_short = merged["target_valid"] & (merged["findings_token_count"] < args.min_tokens)
    too_long = merged["target_valid"] & (merged["findings_token_count"] > upper)
    merged["target_filter_reason"] = np.select(
        [~merged["target_valid"], too_short, too_long],
        ["NO_FINDINGS", "TOO_SHORT", "TOO_LONG"],
        default="VALID",
    )
    merged.loc[too_short | too_long, "target_valid"] = False
    # An invalid target is blanked, not merely masked in metadata, so teacher
    # branches cannot accidentally consume impression/preamble text.
    merged.loc[~merged["target_valid"], "findings_clean"] = ""
    print(train_lengths.describe().to_string())
    print(f"train-derived token bounds: [{args.min_tokens}, {upper}] "
          f"(upper q={args.upper_quantile})")
    print(merged["target_filter_reason"].value_counts().to_string())

    # IMPRESSION is filtered on its own train-derived length distribution.
    # Reusing the FINDINGS bounds would discard most impressions, which are
    # legitimately much shorter (often a single sentence).
    merged["impression_token_count"] = (
        merged["impression_clean"].map(count_lexical_tokens).astype(int)
    )
    impression_studies = merged.drop_duplicates(["subject_id", "study_id"])
    impression_lengths = impression_studies.loc[
        (impression_studies["split"] == "train") & impression_studies["impression_valid"],
        "impression_token_count",
    ]
    if impression_lengths.empty:
        raise ValueError(
            "No valid train IMPRESSION sections; check report parsing before "
            "training a findings_and_impression target."
        )
    impression_upper = int(np.ceil(impression_lengths.quantile(args.upper_quantile)))
    merged.loc[
        merged["impression_valid"]
        & (
            (merged["impression_token_count"] < args.min_tokens)
            | (merged["impression_token_count"] > impression_upper)
        ),
        "impression_valid",
    ] = False
    merged.loc[~merged["impression_valid"], "impression_clean"] = ""
    print(f"impression token bounds: [{args.min_tokens}, {impression_upper}] "
          f"(upper q={args.upper_quantile})")
    print(f"impression usable after length filter: {int(merged['impression_valid'].sum())} rows; "
          f"findings+impression usable: "
          f"{int((merged['target_valid'] & merged['impression_valid']).sum())} rows")

    print("\n=== image_path ===")
    merged["image_path"] = [
        build_image_path(s, st, d)
        for s, st, d in zip(merged["subject_id"], merged["study_id"], merged["dicom_id"])
    ]
    print(f"generated {len(merged)} relative image paths")

    print("\n=== save ===")
    final_cols = ["subject_id", "study_id", "dicom_id", "split", "ViewPosition",
                  "image_path", "findings_clean", "impression_clean",
                  "extraction_method", "target_valid", "impression_valid",
                  "target_filter_reason", "findings_token_count", "findings_word_count",
                  "impression_token_count", "has_chexpert_label"]
    processed = merged[final_cols]

    for split_name, out_name in SPLIT_TO_FILENAME.items():
        subset = processed[processed["split"] == split_name]
        subset.to_csv(out_dir / f"{out_name}.csv", index=False)
        subset.to_parquet(out_dir / f"{out_name}.parquet", index=False)
        print(f"{split_name:9s} -> {out_name}.csv  rows={len(subset):>7d}  "
              f"studies={subset['study_id'].nunique():>7d}")

    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
