#!/usr/bin/env python3
"""Aggregate-only preflight for an attached private MIMIC subset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image


REQUIRED = {
    "subject_id", "study_id", "dicom_id", "image_path", "findings_clean",
    "target_valid", "impression_valid", "has_chexpert_label", "ViewPosition",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(args.dataset_root)
    summary = {"splits": {}}
    randomizer = random.Random(args.seed)
    seen_subjects = {}
    seen_studies = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(root / "processed" / f"{split}.csv")
        missing = sorted(REQUIRED - set(frame.columns))
        if missing:
            raise ValueError(f"{split} is missing columns: {missing}")
        if frame.empty or frame["dicom_id"].duplicated().any():
            raise ValueError(f"{split} is empty or has duplicate DICOM IDs")
        for subject_id in frame["subject_id"].unique():
            if subject_id in seen_subjects:
                raise ValueError("Subject leakage detected")
            seen_subjects[subject_id] = split
        for key in frame[["subject_id", "study_id"]].drop_duplicates().itertuples(index=False, name=None):
            if key in seen_studies:
                raise ValueError("Study leakage detected")
            seen_studies[key] = split
        sample_index = randomizer.randrange(len(frame))
        image_path = root / str(frame.iloc[sample_index]["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError("A deterministic sample image path is absent")
        with Image.open(image_path) as image:
            image.verify()
        summary["splits"][split] = {
            "images": int(len(frame)),
            "subjects": int(frame["subject_id"].nunique()),
            "studies": int(frame[["subject_id", "study_id"]].drop_duplicates().shape[0]),
            "valid_findings": int(frame["target_valid"].astype(bool).sum()),
            "labelled_images": int(frame["has_chexpert_label"].astype(bool).sum()),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
