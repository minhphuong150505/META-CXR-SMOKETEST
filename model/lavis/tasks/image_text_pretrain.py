"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from model.lavis.common.registry import registry
from model.lavis.tasks.base_task import BaseTask
from model.lavis.datasets.data_utils import move_to_cuda
from model.lavis.common.dist_utils import is_dist_avail_and_initialized
import logging
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@registry.register_task("image_text_pretrain_eval")
class ImageTextPretrainTask(BaseTask):
    """Stage-1 validation.

    The confusion-matrix metrics below are kept **bit-identical** to their
    historical behaviour: ``f1_positive_macro`` is the checkpoint selection
    metric, and changing its value would silently change which checkpoint a
    resumed run selects. Its known defect -- a pathology with no positive
    samples contributes 0 to the macro instead of being excluded -- is fixed in
    ``training/evaluation/classification_metrics.py`` and surfaced here under
    separate ``*_defined_only`` keys.

    Set ``run.save_predictions: true`` in the run config to additionally write a
    prediction ``.npz``, which lets the full offline evaluator
    (``scripts/evaluate_stage1.py``) recompute AUROC/AUPRC, calibrate
    thresholds and bootstrap without another GPU pass.
    """

    def __init__(self):
        super().__init__()
        self.run_cfg = None

    @classmethod
    def setup_task(cls, cfg=None, **kwargs):
        task = cls()
        task.run_cfg = cfg.run_cfg if cfg is not None else None
        return task

    def evaluation(self, model, data_loader, cuda_enabled=True):
        loss_sums = {}
        example_count = 0
        confusion = None

        save_predictions = bool(
            self.run_cfg.get("save_predictions", False)
            if self.run_cfg is not None
            else False
        )
        uncertain_policy = str(
            self.run_cfg.get("uncertain_policy", "three_class")
            if self.run_cfg is not None
            else "three_class"
        )
        allowed_policies = {
            "three_class",
            "ignore_uncertain",
            "uncertain_as_positive",
            "uncertain_as_negative",
        }
        if uncertain_policy not in allowed_policies:
            raise ValueError(
                f"Unknown uncertain_policy={uncertain_policy!r}; "
                f"expected one of {sorted(allowed_policies)}"
            )
        classification_only = bool(
            self.run_cfg.get("classification_only_eval", False)
            if self.run_cfg is not None
            else False
        )
        collected_logits = []
        collected_labels = []
        collected_keys = []

        for batch in data_loader:
            if cuda_enabled:
                batch = move_to_cuda(batch)
            if classification_only:
                logits, _ = model.forward_image(batch)
                output = {"classification_logits": logits}
            else:
                output = model(batch)

            labels = batch.get("classification_labels")
            batch_size = labels.shape[0] if labels is not None else len(batch["text_output"])
            example_count += batch_size
            for name, value in output.items():
                if "loss" in name and value is not None:
                    loss_sums[name] = loss_sums.get(name, 0.0) + float(value) * batch_size

            logits = output.get("classification_logits")
            if logits is None or labels is None:
                continue
            if logits.ndim != 3 or labels.shape != logits.shape[:2]:
                raise ValueError(
                    "classification logits/labels must be [B, abnormalities, classes] "
                    f"and [B, abnormalities], got {tuple(logits.shape)} and {tuple(labels.shape)}"
                )

            sample_mask = output.get("classification_mask")
            if sample_mask is None:
                sample_mask = batch.get("classification_mask")
            if sample_mask is None:
                sample_mask = batch.get("has_chexpert_label")
            if sample_mask is None:
                sample_mask = torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
            sample_mask = torch.as_tensor(
                sample_mask, dtype=torch.bool, device=labels.device
            ).reshape(-1)

            if classification_only:
                # `forward_image` is the image-only student path and returns no
                # loss, so with classification_only_eval the run had NO validation
                # loss at all -- nothing for `selection_metric: loss_cls` to
                # select on. Recompute it here with the same ClassificationLoss
                # and the same per-sample mask the training step uses, so val and
                # train report the same quantity.
                loss_cls = model.cls_loss_fn(logits, labels, sample_mask=sample_mask)
                loss_sums["loss_cls"] = (
                    loss_sums.get("loss_cls", 0.0) + float(loss_cls) * batch_size
                )

            metric_labels = labels.clone()
            label_mask = sample_mask[:, None] & (metric_labels >= 0)
            if uncertain_policy == "ignore_uncertain":
                label_mask &= metric_labels != 2
            elif uncertain_policy == "uncertain_as_positive":
                metric_labels[metric_labels == 2] = 1
            elif uncertain_policy == "uncertain_as_negative":
                metric_labels[metric_labels == 2] = 0

            num_abnormalities, num_classes = logits.shape[1:]
            if confusion is None:
                confusion = torch.zeros(
                    num_abnormalities,
                    num_classes,
                    num_classes,
                    dtype=torch.float64,
                    device=logits.device,
                )
            predictions = logits.argmax(dim=-1)
            valid = label_mask & (metric_labels < num_classes)
            abnormality_idx = torch.arange(num_abnormalities, device=labels.device)[None, :]
            flat_index = (
                abnormality_idx * num_classes * num_classes
                + metric_labels.long() * num_classes
                + predictions.long()
            )
            counts = torch.bincount(
                flat_index[valid], minlength=num_abnormalities * num_classes * num_classes
            )
            confusion += counts.reshape(num_abnormalities, num_classes, num_classes)

            if save_predictions:
                # Labels are masked to -1 where the sample carries no CheXpert
                # annotation, so the offline evaluator skips exactly the rows
                # this loop skipped. Without it, an unlabelled row would be
                # read back as a true negative.
                masked_labels = metric_labels.clone()
                masked_labels[~valid] = -1
                collected_logits.append(logits.detach().float().cpu())
                collected_labels.append(masked_labels.detach().cpu())
                collected_keys.extend(_sample_keys(batch, labels.shape[0]))

        device = next(model.parameters()).device
        loss_names = sorted(loss_sums)
        totals = torch.tensor(
            [loss_sums[name] for name in loss_names] + [example_count],
            dtype=torch.float64, device=device,
        )
        if is_dist_avail_and_initialized():
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            if confusion is not None:
                dist.all_reduce(confusion, op=dist.ReduceOp.SUM)

        denom = max(totals[-1].item(), 1.0)
        stats = {
            name: totals[index].item() / denom
            for index, name in enumerate(loss_names)
        }

        if confusion is not None:
            true_positive = confusion.diagonal(dim1=1, dim2=2)
            support = confusion.sum(dim=2)
            predicted = confusion.sum(dim=1)
            precision = true_positive / predicted.clamp_min(1.0)
            recall = true_positive / support.clamp_min(1.0)
            f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
            stats.update(
                {
                    "precision_macro": precision.mean().item(),
                    "recall_macro": recall.mean().item(),
                    "f1_macro": f1.mean().item(),
                    "f1_weighted": (
                        (f1 * support).sum() / support.sum().clamp_min(1.0)
                    ).item(),
                    "accuracy": (
                        true_positive.sum() / confusion.sum().clamp_min(1.0)
                    ).item(),
                }
            )
            # CheXpert class index 1 is the positive finding class. Reporting
            # this separately avoids a deceptively high F1 dominated by common
            # negatives and is the primary classification-quality signal.
            if confusion.shape[-1] > 1:
                stats.update(
                    {
                        "precision_positive_macro": precision[:, 1].mean().item(),
                        "recall_positive_macro": recall[:, 1].mean().item(),
                        "f1_positive_macro": f1[:, 1].mean().item(),
                    }
                )

                # Same quantities, averaged over pathologies that actually have
                # positive samples in this split. The keys above dilute the
                # macro with a hard 0 for every pathology with zero positive
                # support, which makes them shift with split composition; these
                # do not. Reported alongside rather than instead, so the
                # selection metric and every historical log stay comparable.
                has_positive = support[:, 1] > 0
                num_defined = int(has_positive.sum().item())
                if num_defined:
                    stats.update(
                        {
                            "precision_positive_macro_defined_only": precision[has_positive, 1].mean().item(),
                            "recall_positive_macro_defined_only": recall[has_positive, 1].mean().item(),
                            "f1_positive_macro_defined_only": f1[has_positive, 1].mean().item(),
                            "num_pathologies_with_positives": float(num_defined),
                        }
                    )
                if num_defined < confusion.shape[0]:
                    logger.info(
                        "%d/%d pathologies have no positive samples in this split; "
                        "they drag f1_positive_macro down but are excluded from "
                        "f1_positive_macro_defined_only",
                        confusion.shape[0] - num_defined,
                        confusion.shape[0],
                    )

        if save_predictions and collected_logits:
            self._save_predictions(
                collected_logits, collected_labels, collected_keys, model
            )

        return stats

    @staticmethod
    def _save_predictions(logits_chunks, label_chunks, keys, model):
        """Write a prediction .npz for offline re-evaluation.

        Best-effort: a failure here must not lose a completed validation pass,
        so it is logged rather than raised.
        """
        try:
            import numpy as np

            from model.lavis.models.blip2_models.blip2_qformer import chexpert_cols
            from training.evaluation.schemas import ClassificationPredictions

            if is_dist_avail_and_initialized() and dist.get_world_size() > 1:
                logger.warning(
                    "save_predictions collects rank-local batches only; with "
                    "world_size>1 the written file covers this rank's shard"
                )

            logits = torch.cat(logits_chunks, dim=0).numpy()
            labels = torch.cat(label_chunks, dim=0).numpy()
            probabilities = torch.softmax(
                torch.from_numpy(logits), dim=-1
            ).numpy()

            names = tuple(chexpert_cols)
            if len(names) != labels.shape[1]:
                names = tuple(f"abnormality_{i}" for i in range(labels.shape[1]))
                logger.warning(
                    "chexpert_cols has %d entries but logits have %d columns; "
                    "falling back to positional names",
                    len(chexpert_cols),
                    labels.shape[1],
                )

            predictions = ClassificationPredictions(
                labels=labels,
                probabilities=probabilities,
                logits=logits,
                pathology_names=names,
                sample_keys=np.asarray(keys[: labels.shape[0]], dtype="U"),
            )
            registry_output = registry.get_path("result_dir") or "."
            path = f"{registry_output}/validation_predictions.npz"
            predictions.save(path)
            logger.info("saved %d predictions to %s", labels.shape[0], path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not save predictions (metrics unaffected): %s", exc)


def _sample_keys(batch, batch_size):
    """Opaque per-study keys, preferring an identifier the batch already carries."""
    for field in ("study_id", "image_id", "dicom_id"):
        values = batch.get(field)
        if values is None:
            continue
        if torch.is_tensor(values):
            values = values.detach().cpu().tolist()
        return [str(v) for v in list(values)[:batch_size]]
    return [""] * batch_size
