#!/usr/bin/env python3
"""Generate held-out FINDINGS reports and compute aggregate NLP metrics only."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.lavis.common.config import Config
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset
from model.lavis.tasks.image_text_pretrain import ImageTextPretrainTask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--min-length", type=int, default=8)
    parser.add_argument("--bertscore-model", default="roberta-large")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, 1):
            current.append(previous[index - 1] + 1 if token == other else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def compute_metrics(references: list[str], hypotheses: list[str], bertscore_model: str) -> dict:
    """Compute corpus metrics without retaining protected report text on disk."""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
        from nltk.translate.meteor_score import meteor_score
        from rouge_score.rouge_scorer import RougeScorer
        from pycocoevalcap.cider.cider import Cider
        from bert_score import score as bertscore
    except ImportError as exc:
        raise RuntimeError(
            "Missing NLP metric dependency. Install nltk, rouge-score, "
            "pycocoevalcap and bert-score before evaluation."
        ) from exc

    reference_tokens = [[_tokens(text)] for text in references]
    hypothesis_tokens = [_tokens(text) for text in hypotheses]
    smoother = SmoothingFunction().method1
    rouge = RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = [rouge.score(ref, hyp)["rougeL"].fmeasure for ref, hyp in zip(references, hypotheses, strict=True)]
    meteor = [meteor_score([ref], hyp) for ref, hyp in zip(reference_tokens, hypothesis_tokens, strict=True)]
    cider, _ = Cider().compute_score(
        {index: [reference] for index, reference in enumerate(references)},
        {index: [hypothesis] for index, hypothesis in enumerate(hypotheses)},
    )
    _, _, bert_f1 = bertscore(hypotheses, references, model_type=bertscore_model, lang="en", verbose=True)
    return {
        "bleu_1": float(corpus_bleu(reference_tokens, hypothesis_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother)),
        "bleu_4": float(corpus_bleu(reference_tokens, hypothesis_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)),
        "rouge_l": float(np.mean(rouge_l)),
        "meteor": float(np.mean(meteor)),
        "cider": float(cider),
        "bertscore_f1": float(bert_f1.mean().item()),
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}; pass --overwrite to replace it")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("Report evaluation requires one CUDA GPU")

    cfg = Config(SimpleNamespace(cfg_path=args.cfg_path, options=[
        "run.evaluate=true", "run.distributed=false", "run.world_size=1",
        "run.test_splits=[test]", "run.valid_splits=[]",
    ]))
    model = ImageTextPretrainTask().build_model(cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected_identity = {
        "dataset_manifest_sha256": args.dataset_manifest_sha256,
        "config_fingerprint": args.config_fingerprint,
    }
    if checkpoint.get("identity") != expected_identity:
        raise RuntimeError("Checkpoint identity mismatch")
    result = model.load_state_dict(checkpoint["model"], strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {result.unexpected_keys[:8]}")
    model.to("cuda:0").eval()

    from local_config import VIS_ROOT
    dataset = MIMIC_CXR_Dataset(vis_processor=None, text_processor=None, vis_root=VIS_ROOT, split="test", cfg=cfg)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, collate_fn=dataset.collater)
    references, hypotheses = [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            for key, value in list(batch.items()):
                if torch.is_tensor(value):
                    batch[key] = value.to("cuda:0", non_blocking=True)
            generated = model.generate(batch, num_beams=args.num_beams, max_length=args.max_length, min_length=args.min_length)
            valid = torch.as_tensor(batch.get("generation_mask", True)).detach().cpu().bool().tolist()
            for reference, hypothesis, keep in zip(batch["text_output"], generated, valid, strict=True):
                if keep:
                    references.append(reference)
                    hypotheses.append(hypothesis)
    if not references:
        raise RuntimeError("No valid test FINDINGS targets")
    metrics = compute_metrics(references, hypotheses, args.bertscore_model)
    payload = {
        "status": "pass", "method": "test_set_report_generation_aggregate_only",
        "test_studies": len(dataset), "evaluated_reports": len(references),
        "checkpoint_identity": expected_identity, "checkpoint_epoch": checkpoint.get("epoch"),
        "generation": {"num_beams": args.num_beams, "max_length": args.max_length, "min_length": args.min_length},
        "metrics": metrics, "bertscore_model": args.bertscore_model,
        "wall_seconds": time.perf_counter() - started,
        "privacy": "Aggregate metrics only; generated reports and references are not written.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
