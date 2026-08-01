#!/usr/bin/env python3
"""Evaluate the six encoder configurations in paper Table 5 from one E123 checkpoint.

All three frozen encoders run once per held-out batch. Inactive shared-token
spans are removed before MHCAC for each subset. Results are sensitivity
measurements, not independently trained ablations or causal contributions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.lavis.common.config import Config
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
from model.lavis.models.blip2_models.blip2_qformer import Blip2Qformer, chexpert_cols
from model.lavis.tasks.image_text_pretrain import ImageTextPretrainTask
from training.evaluation.classification_metrics import evaluate_classification
from training.evaluation.paper_table5 import (
    PAPER_TABLE5_PATHOLOGIES,
    paper_table5_weighted_f1,
)
from training.evaluation.schemas import ClassificationPredictions


MASKS = {
    "E1": ("biovil",),
    "E2": ("pubmedclip",),
    "E3": ("swin",),
    "E12": ("biovil", "pubmedclip"),
    "E13": ("biovil", "swin"),
    "E123": ("biovil", "pubmedclip", "swin"),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing aggregate-only sensitivity result",
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def load_model_state(model, checkpoint_path: Path, expected_identity: dict) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("identity") != expected_identity:
        raise RuntimeError(
            f"Checkpoint identity mismatch: expected {expected_identity}, "
            f"found {checkpoint.get('identity')}"
        )
    state = checkpoint["model"]
    model_state = model.state_dict()
    mismatched = [
        name for name, value in state.items()
        if name in model_state and tuple(value.shape) != tuple(model_state[name].shape)
    ]
    if mismatched:
        raise RuntimeError(f"Checkpoint tensor shapes mismatch: {mismatched[:8]}")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {result.unexpected_keys[:8]}")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    missing_trainable = sorted(trainable.intersection(result.missing_keys))
    if missing_trainable:
        raise RuntimeError(f"Missing trainable checkpoint keys: {missing_trainable[:8]}")
    return checkpoint


def json_safe(value):
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def sensitivity_deltas(scores: dict[str, float]) -> dict:
    """Return the declared pair/singleton comparisons for one metric."""
    return {
        "E123_minus_pairs": {
            pair: scores["E123"] - scores[pair]
            for pair in ("E12", "E13")
        },
        "pairs_minus_singletons": {
            "E12-E1": scores["E12"] - scores["E1"],
            "E12-E2": scores["E12"] - scores["E2"],
            "E13-E1": scores["E13"] - scores["E1"],
            "E13-E3": scores["E13"] - scores["E3"],
        },
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Refusing to rerun held-out sensitivity evaluation over existing {output}"
            )
        output.unlink()
        output.with_suffix(".md").unlink(missing_ok=True)
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Sensitivity evaluation requires one CUDA GPU")

    options = [
        "run.evaluate=true",
        "run.distributed=false",
        "run.world_size=1",
        "run.test_splits=[test]",
        "run.valid_splits=[]",
    ]
    cfg = Config(SimpleNamespace(cfg_path=args.cfg_path, options=options))
    model = ImageTextPretrainTask().build_model(cfg)
    # The eval identity gates ONLY on dataset + config fingerprint, matching the
    # two-field dict training writes into checkpoint["identity"]. source_commit
    # is provenance (stored top-level, deliberately NOT in identity), so the
    # eval-code commit may differ from the commit that trained the checkpoint
    # without stranding it.
    expected_identity = {
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "config_fingerprint": args.config_fingerprint,
    }
    checkpoint = load_model_state(model, Path(args.checkpoint), expected_identity)
    checkpoint_source_commit = checkpoint.get("source_commit")
    device = torch.device("cuda:0")
    model.to(device).eval()

    from local_config import VIS_ROOT

    dataset = MIMIC_CXR_Dataset(
        vis_processor=None, text_processor=None, vis_root=VIS_ROOT,
        split="test", cfg=cfg,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=dataset.collater,
    )

    logits_by_mask = {name: [] for name in MASKS}
    labels_chunks = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch in loader:
            for key, value in list(batch.items()):
                if torch.is_tensor(value):
                    batch[key] = value.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                shared = model.encode_visual_once(batch)
                for name, keep in MASKS.items():
                    logits_by_mask[name].append(
                        model.classify_shared_visual(shared, keep).float().cpu()
                    )
            labels = batch["classification_labels"].detach().cpu().clone()
            sample_mask = batch.get("classification_mask", batch.get("has_chexpert_label"))
            if sample_mask is not None:
                labels[~torch.as_tensor(sample_mask).detach().cpu().bool()] = -1
            labels_chunks.append(labels)

    labels = torch.cat(labels_chunks).numpy()
    reports = {}
    for name, chunks in logits_by_mask.items():
        logits = torch.cat(chunks).numpy()
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=-1, keepdims=True)
        predictions = ClassificationPredictions(
            labels=labels,
            probabilities=probabilities,
            logits=None,
            pathology_names=tuple(chexpert_cols),
            sample_keys=np.arange(labels.shape[0]).astype(str),
            metadata={"encoder_keep": list(MASKS[name])},
        )
        binary = (logits.argmax(axis=-1) == 1).astype(np.int64)
        reports[name] = evaluate_classification(
            predictions,
            uncertain_policy="ignore_uncertain",
            include_meta_labels=False,
            binary_predictions=binary,
        ).to_dict()
        reports[name]["paper_table5"] = paper_table5_weighted_f1(
            labels, logits, tuple(chexpert_cols)
        )

    metric = "positive_macro_f1"
    score = {name: reports[name]["aggregates"][metric] for name in MASKS}
    paper_score = {
        name: reports[name]["paper_table5"]["mean_weighted_f1"] for name in MASKS
    }
    deltas = {
        "paper_table5_weighted_f1": sensitivity_deltas(paper_score),
        "positive_macro_f1": sensitivity_deltas(score),
    }
    payload = {
        "status": "pass",
        "method": "single_E123_checkpoint_post_training_inference_sensitivity",
        "warning": "Not independent ablation and not causal contribution.",
        "standard_report_uncertain_policy": "ignore_uncertain",
        "checkpoint_identity": expected_identity,
        "checkpoint_source_commit": checkpoint_source_commit,
        "eval_source_commit": args.source_commit,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "masks": {name: list(value) for name, value in MASKS.items()},
        "paper_table5_protocol": {
            "metric": "mean_per_pathology_weighted_f1_three_class_argmax",
            "pathologies": list(PAPER_TABLE5_PATHOLOGIES),
            "classes": ["negative", "positive", "uncertain"],
            "encoder_selection": "remove_inactive_token_spans_before_mhcac",
        },
        "reports": reports,
        "deltas": deltas,
        "test_studies": len(dataset),
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True), encoding="utf-8")

    table = [
        "# Encoder inference sensitivity", "",
        "> One E123 training run; selective encoder activation at inference, matching the paper's Table-5 ablation protocol.", "",
        "| ID | Encoders kept | Paper Table-5 weighted F1 | Positive macro F1 |",
        "|---|---|---:|---:|",
    ]
    for name, keep in MASKS.items():
        table.append(
            f"| {name} | {' + '.join(keep)} | {paper_score[name]:.6f} | {score[name]:.6f} |"
        )
    output.with_suffix(".md").write_text("\n".join(table) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
