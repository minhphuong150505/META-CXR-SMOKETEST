"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import datetime
import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from model.lavis.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_dist_avail_and_initialized,
    is_main_process,
    main_process,
)
from model.lavis.common.registry import registry
from model.lavis.common.utils import is_url
from model.lavis.datasets.data_utils import concat_datasets, reorg_datasets_by_split
from model.lavis.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ChainDataset

from torchinfo import summary
from smoke.identity import assert_checkpoint_identity, build_checkpoint_identity
from smoke.samplers import DistributedEvalSampler


def _state_dict_has_non_finite(state_dict):
    """Recursively scan a (nested) state_dict-like object for non-finite floats.

    Returns (has_bad: bool, count: int). Catches NaN/Inf in Adam momentum
    buffers, scaler scale, or model weights that would otherwise propagate
    silently on load and corrupt a resumed run.
    """
    bad = 0

    def walk(obj):
        nonlocal bad
        if isinstance(obj, torch.Tensor) and obj.is_floating_point():
            if not torch.isfinite(obj).all():
                bad += 1
        elif isinstance(obj, dict):
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                walk(v)

    walk(state_dict)
    return bad > 0, bad


def _scaler_state_is_degenerate(scaler_state):
    """Detect a GradScaler state that has degraded to scale<=0 (training-dead).

    After enough consecutive AMP overflows, scale halves down to fp32 underflow
    and becomes exactly 0. From that point: loss*0=0, grad=0, then unscale_
    divides by 0 producing NaN — every step is skipped forever. Loading such a
    scaler state poisons the resumed run even though no tensor is NaN/Inf.
    """
    if not isinstance(scaler_state, dict):
        return False, None
    scale = scaler_state.get("scale")
    if scale is None:
        return False, None
    try:
        scale_val = scale.item() if isinstance(scale, torch.Tensor) else float(scale)
    except (TypeError, ValueError):
        return False, None
    if not (scale_val > 0):
        return True, scale_val
    return False, scale_val


@registry.register_runner("runner_base")
class RunnerBase:
    """
    A runner class to train and evaluate a model given a task and datasets.

    The runner uses pytorch distributed data parallel by default. Future release
    will support other distributed frameworks.
    """

    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.job_id = job_id
        self.run_name = self.config.run_cfg.run_name

        self.task = task
        self.datasets = datasets
        if 'coco_caption' in self.datasets:
            self.gt_map = {}
            for split_name in ["train", "val", "test"]:
                self.gt_map[split_name] = {int(elem['image'].split(".")[0].split("_")[-1]): "/".join(elem['caption']) if split_name != "train" else elem['caption'] for elem in self.datasets['coco_caption'][split_name].annotation}

        self._model = model

        self._wrapped_model = None
        self._device = None
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

        self.start_epoch = 0

        # self.setup_seeds()
        self.setup_output_dir()

    def generate_html_table(self, data, columns):
        html = '<table border="1" cellpadding="5" cellspacing="0">'
        html += "<tr>"
        for col in columns:
            html += f"<th>{col}</th>"
        html += "</tr>"

        for row in data:
            html += "<tr>"
            for cell in row:
                html += f"<td>{cell}</td>"
            html += "</tr>"
        html += "</table>"

        return html

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(self.config.run_cfg.device)

        return self._device

    @property
    def use_distributed(self):
        return self.config.run_cfg.distributed

    @property
    def model(self):
        """
        A property to get the DDP-wrapped model on the device.
        """
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)

            # distributed training wrapper
            if self.use_distributed:
                if self._wrapped_model is None:
                    self._wrapped_model = DDP(
                        self._model, device_ids=[self.config.run_cfg.gpu],
                        find_unused_parameters=bool(
                            self.config.run_cfg.get("find_unused_parameters", False)
                        ),
                    )
            else:
                self._wrapped_model = self._model

        return self._wrapped_model

    @property
    def optimizer(self):
        # TODO make optimizer class and configurations
        if self._optimizer is None:
            num_parameters = 0
            grouped_params = {
                "classifier_decay": [],
                "classifier_no_decay": [],
                "qformer_decay": [],
                "qformer_no_decay": [],
            }
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue  # frozen weights
                is_classifier = any(
                    token in n
                    for token in ("mhcac", "aggregator", "cls_loss_fn")
                )
                no_decay = p.ndim < 2 or n.endswith(".bias") or any(
                    token in n.lower() for token in ("layernorm", "layer_norm", ".ln", ".bn")
                )
                prefix = "classifier" if is_classifier else "qformer"
                suffix = "no_decay" if no_decay else "decay"
                grouped_params[f"{prefix}_{suffix}"].append(p)
                num_parameters += p.data.nelement()
            total_parameters = sum(p.numel() for p in self.model.parameters())
            logging.info(
                "parameters: trainable=%d total=%d frozen=%d",
                num_parameters,
                total_parameters,
                total_parameters - num_parameters,
            )
            base_lr = float(self.config.run_cfg.init_lr)
            if base_lr <= 0:
                raise ValueError("run.init_lr must be positive")
            cls_lr = float(self.config.run_cfg.get("init_lr_cls", base_lr))
            qformer_lr = float(self.config.run_cfg.get("init_lr_q", base_lr))
            weight_decay = float(self.config.run_cfg.weight_decay)
            optim_params = []
            for name, params in grouped_params.items():
                if not params:
                    continue
                group_lr = cls_lr if name.startswith("classifier") else qformer_lr
                optim_params.append(
                    {
                        "name": name,
                        "params": params,
                        "weight_decay": 0.0 if name.endswith("no_decay") else weight_decay,
                        "lr": group_lr,
                        "lr_scale": group_lr / base_lr,
                    }
                )
            beta2 = self.config.run_cfg.get("beta2", 0.999)
            self._optimizer = torch.optim.AdamW(
                optim_params,
                lr=base_lr,
                weight_decay=weight_decay,
                betas=(0.9, beta2),
            )

        return self._optimizer

    @property
    def scaler(self):
        amp = self.config.run_cfg.get("amp", False)

        if amp:
            if self._scaler is None:
                try:
                    self._scaler = torch.amp.GradScaler("cuda")
                except (AttributeError, TypeError):
                    self._scaler = torch.cuda.amp.GradScaler()

        return self._scaler

    @property
    def lr_scheduler(self):
        """
        A property to get and create learning rate scheduler by split just in need.
        """
        if self._lr_sched is None:
            lr_sched_cls = registry.get_lr_scheduler_class(self.config.run_cfg.lr_sched)

            # max_epoch = self.config.run_cfg.max_epoch
            max_epoch = int(
                self.config.run_cfg.get("scheduler_max_epoch", self.max_epoch)
            )
            # min_lr = self.config.run_cfg.min_lr
            min_lr = self.min_lr
            # init_lr = self.config.run_cfg.init_lr
            init_lr = self.init_lr

            # optional parameters
            decay_rate = self.config.run_cfg.get("lr_decay_rate", 1.)
            warmup_start_lr = self.config.run_cfg.get("warmup_lr", -1)
            warmup_steps = self.config.run_cfg.get("warmup_steps", 0)

            self._lr_sched = lr_sched_cls(
                optimizer=self.optimizer,
                max_epoch=max_epoch,
                min_lr=min_lr,
                init_lr=init_lr,
                decay_rate=decay_rate,
                warmup_start_lr=warmup_start_lr,
                warmup_steps=warmup_steps,
            )

        return self._lr_sched

    @property
    def dataloaders(self) -> dict:
        """
        A property to get and create dataloaders by split just in need.

        If no train_dataset_ratio is provided, concatenate map-style datasets and
        chain wds.DataPipe datasets separately. Training set becomes a tuple
        (ConcatDataset, ChainDataset), both are optional but at least one of them is
        required. The resultant ConcatDataset and ChainDataset will be sampled evenly.

        If train_dataset_ratio is provided, create a MultiIterLoader to sample
        each dataset by ratios during training.

        Currently do not support multiple datasets for validation and test.

        Returns:
            dict: {split_name: (tuples of) dataloader}
        """
        if self._dataloaders is None:
            # reoganize datasets by split and concatenate/chain if necessary
            dataset_ratios = self.config.run_cfg.get("train_dataset_ratios", None)

            # concatenate map-style datasets and chain wds.DataPipe datasets separately
            # training set becomes a tuple (ConcatDataset, ChainDataset), both are
            # optional but at least one of them is required. The resultant ConcatDataset
            # and ChainDataset will be sampled evenly.
            logging.info(
                "dataset_ratios not specified, datasets will be concatenated (map-style datasets) or chained (webdataset.DataPipeline)."
            )

            datasets = reorg_datasets_by_split(self.datasets)
            #self.datasets = concat_datasets(datasets)
            # select first dataset for validation and test
            self.datasets = {k: v[0] for k, v in datasets.items()}

            # print dataset statistics after concatenation/chaining
            for split_name in self.datasets:
                if isinstance(self.datasets[split_name], tuple) or isinstance(
                    self.datasets[split_name], list
                ):
                    # mixed wds.DataPipeline and torch.utils.data.Dataset
                    num_records = sum(
                        [
                            len(d)
                            if not type(d) in [ChainDataset]
                            else 0
                            for d in self.datasets[split_name]
                        ]
                    )

                else:
                    if hasattr(self.datasets[split_name], "__len__"):
                        # a single map-style dataset
                        num_records = len(self.datasets[split_name])
                    else:
                        # a single wds.DataPipeline
                        num_records = -1
                        logging.info(
                            "Only a single wds.DataPipeline dataset, no __len__ attribute."
                        )

                if num_records >= 0:
                    logging.info(
                        "Loaded {} records for {} split from the dataset.".format(
                            num_records, split_name
                        )
                    )

            # create dataloaders
            split_names = sorted(self.datasets.keys())

            datasets = [self.datasets[split] for split in split_names]
            is_trains = [split in self.train_splits for split in split_names]

            batch_sizes = [
                self.config.run_cfg.batch_size_train
                if split == "train"
                else self.config.run_cfg.batch_size_eval
                for split in split_names
            ]

            collate_fns = []
            for dataset in datasets:
                if isinstance(dataset, tuple) or isinstance(dataset, list):
                    collate_fns.append([getattr(d, "collater", None) for d in dataset])
                else:
                    collate_fns.append(getattr(dataset, "collater", None))

            dataloaders = self.create_loaders(
                datasets=datasets,
                num_workers=self.config.run_cfg.num_workers,
                batch_sizes=batch_sizes,
                is_trains=is_trains,
                collate_fns=collate_fns,
                dataset_ratios=dataset_ratios,
            )

            self._dataloaders = {k: v for k, v in zip(split_names, dataloaders)}

        return self._dataloaders

    @property
    def cuda_enabled(self):
        return self.device.type == "cuda"

    @property
    def max_epoch(self):
        return int(self.config.run_cfg.max_epoch)

    @property
    def log_freq(self):
        log_freq = self.config.run_cfg.get("log_freq", 50)
        return int(log_freq)

    @property
    def init_lr(self):
        return float(self.config.run_cfg.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run_cfg.min_lr)

    @property
    def accum_grad_iters(self):
        return int(self.config.run_cfg.get("accum_grad_iters", 1))

    @property
    def valid_splits(self):
        valid_splits = self.config.run_cfg.get("valid_splits", [])

        if len(valid_splits) == 0:
            logging.info("No validation splits found.")

        return valid_splits

    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])

        return test_splits

    @property
    def train_splits(self):
        train_splits = self.config.run_cfg.get("train_splits", [])

        if len(train_splits) == 0:
            logging.info("Empty train splits.")

        return train_splits

    @property
    def evaluate_only(self):
        """
        Set to True to skip training.
        """
        return self.config.run_cfg.evaluate

    @property
    def use_dist_eval_sampler(self):
        return self.config.run_cfg.get("use_dist_eval_sampler", True)

    @property
    def resume_ckpt_path(self):
        return self.config.run_cfg.get("resume_ckpt_path", None)

    @property
    def save_freq(self):
        return int(self.config.run_cfg.get("save_freq", 1))

    @property
    def max_grad_norm(self):
        return float(self.config.run_cfg.get("max_grad_norm", 1.0))

    @property
    def early_stop_patience(self):
        return int(self.config.run_cfg.get("early_stop_patience", -1))

    @property
    def early_stop_min_delta(self):
        return float(self.config.run_cfg.get("early_stop_min_delta", 0.0))

    @property
    def selection_metric(self):
        metric = self.config.run_cfg.get("selection_metric")
        if not metric:
            raise ValueError("run.selection_metric must be set explicitly")
        return str(metric)

    @property
    def selection_mode(self):
        configured = self.config.run_cfg.get("selection_mode", None)
        if configured is None:
            configured = "min" if "loss" in self.selection_metric else "max"
        configured = str(configured).lower()
        if configured not in {"min", "max"}:
            raise ValueError("run.selection_mode must be either 'min' or 'max'")
        return configured

    def _metric_improved(self, value, best):
        if self.selection_mode == "min":
            return value < best - self.early_stop_min_delta
        return value > best + self.early_stop_min_delta

    @property
    def train_loader(self):
        train_dataloader = self.dataloaders["train"]

        return train_dataloader

    def setup_output_dir(self):
        base_dir = Path(self.config.run_cfg.get("output_dir", "pretraining/outputs"))

        output_dir = base_dir if base_dir.name == self.run_name else base_dir / self.run_name
        if self.evaluate_only:
            output_dir = base_dir / self.run_name.replace("_eval", "")
        result_dir = output_dir / "result"

        # DDP-safe: only the main process runs the "refuse to overwrite" check
        # and creates the directories, then the other ranks wait on a barrier.
        # If every rank did this, a slower rank would see the directory the
        # faster rank just created (via the mkdir below) as non-empty and raise
        # a spurious FileExistsError -- the race that made exactly one of the two
        # ranks fail on every launch, regardless of whether the dir was clean.
        if is_main_process():
            if (
                not self.evaluate_only
                and output_dir.exists()
                and any(output_dir.iterdir())
                and not self.resume_ckpt_path
            ):
                raise FileExistsError(
                    f"Refusing to overwrite non-empty run directory {output_dir}; "
                    "set run.resume_ckpt_path to resume this exact run."
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            result_dir.mkdir(parents=True, exist_ok=True)

        if is_dist_avail_and_initialized():
            dist.barrier()

        registry.register_path("result_dir", str(result_dir))
        registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def _checkpoint_identity(self):
        return build_checkpoint_identity(self.config.run_cfg)

    def _load_trainable_state(self, model, state_dict, checkpoint_path):
        model_state = model.state_dict()
        mismatched = [
            (key, tuple(value.shape), tuple(model_state[key].shape))
            for key, value in state_dict.items()
            if key in model_state
            and hasattr(value, "shape")
            and tuple(value.shape) != tuple(model_state[key].shape)
        ]
        if mismatched:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has tensor shape mismatches: "
                f"{mismatched[:8]}"
            )
        result = model.load_state_dict(state_dict, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} has unexpected keys: "
                f"{result.unexpected_keys[:8]}"
            )
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        missing_trainable = sorted(trainable.intersection(result.missing_keys))
        if missing_trainable:
            raise RuntimeError(
                f"Checkpoint {checkpoint_path} is missing trainable tensors: "
                f"{missing_trainable[:8]}"
            )

    def _reduce_eval_stats(self, eval_stats):
        """Average eval stats across DDP ranks.

        Each rank evaluates only its own shard of the split, so the loss it
        computes is rank-local. Checkpoint selection and early stopping both
        branch on that value, so without this reduction the ranks can disagree
        on when to stop -- one breaks out of the training loop while the others
        block forever on the next collective.

        DistributedSampler pads every rank to the same length, so a plain mean
        is the correct aggregate.
        """
        if not (self.use_distributed and dist.is_available() and dist.is_initialized()):
            return eval_stats

        # Sorted keys so every rank packs the tensor in the same order.
        keys = sorted(eval_stats.keys())
        values = torch.tensor(
            [eval_stats[k] for k in keys], dtype=torch.float64, device=self.device
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= get_world_size()

        return dict(zip(keys, (float(v) for v in values.tolist())))

    def validate(self, cur_epoch, best_agg_metric, best_epoch, wandb_run):
        # The test set is held out until training and checkpoint selection are
        # complete.  Evaluate-only runs may explicitly request test splits.
        requested_splits = self.test_splits if self.evaluate_only else self.valid_splits
        eval_splits = []
        for split_name in requested_splits:
            if split_name not in eval_splits:
                eval_splits.append(split_name)

        if len(eval_splits) > 0:
            for split_name in eval_splits:
                logging.info("Evaluating on {}.".format(split_name))

                val_log, results, gts, loss = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch
                )
                if self.evaluate_only and self.config.run_cfg.get(
                    "save_text_predictions", False
                ):
                    # save free-text predictions and corresponding gt
                    # create folder
                    if not os.path.exists(self.output_dir / "predictions"):
                        os.makedirs(self.output_dir / "predictions")
                        os.makedirs(self.output_dir / "ground_truths")

                    # save predictions
                    with open(self.output_dir / "predictions" / "predictions_{}.txt".format(split_name), "w") as f:
                        for i in range(len(results)):
                            f.write('"' + results[i]['caption'] + '"\n')
                    # save ground truths
                    with open(self.output_dir / "ground_truths" / "ground_truths_{}.txt".format(split_name), "w") as f:
                        for i in [elem['image_id'] for elem in results]:
                            f.write('"' + gts[i][0] + '"\n')

                if val_log is not None:
                    if is_main_process():
                        assert (
                                "agg_metrics" in val_log
                        ), "No agg_metrics found in validation log."

                        agg_metrics = val_log["agg_metrics"]
                        if not self.evaluate_only and split_name == "val": #dont save for train_val
                            if self._metric_improved(agg_metrics, best_agg_metric):
                                best_epoch, best_agg_metric = cur_epoch, agg_metrics
                                self._save_checkpoint(cur_epoch, is_best=True,
                                                      best_agg_metric=best_agg_metric,
                                                      best_epoch=best_epoch)
                                logging.info("Saving best model at epoch {}".format(cur_epoch))

                        val_log.update({"best_epoch": best_epoch})
                        self.log_stats(val_log, split_name)

                        # Never send MIMIC report text or predictions to W&B.
                        # Numeric metrics are logged below; text artifacts stay
                        # within explicitly configured private storage.

                if loss is not None:
                    if isinstance(loss, dict):
                        eval_stats = {k: float(v) for k, v in loss.items()}
                    else:
                        eval_stats = {"loss": float(loss)}

                    # Must run on every rank before the value drives checkpoint
                    # selection and early stopping below.
                    eval_stats = self._reduce_eval_stats(eval_stats)

                    self.log_stats(eval_stats, split_name)

                    selection_value = eval_stats.get(self.selection_metric)
                    if not self.evaluate_only:
                        if (
                            selection_value is not None
                            and self._metric_improved(selection_value, best_agg_metric)
                            and split_name == "val"
                        ):
                            best_epoch, best_agg_metric = cur_epoch, selection_value
                            self._save_checkpoint(cur_epoch, is_best=True,
                                                  best_agg_metric=best_agg_metric,
                                                  best_epoch=best_epoch)
                            logging.info(
                                "Saving best model at epoch %d (val %s %.6f)",
                                cur_epoch,
                                self.selection_metric,
                                selection_value,
                            )
                        elif selection_value is None and split_name == "val":
                            raise KeyError(
                                f"selection metric '{self.selection_metric}' is absent; "
                                f"available metrics: {sorted(eval_stats)}"
                            )

        # Always save `checkpoint_last.pth` at end of every epoch (resume anchor).
        if not self.evaluate_only:
            self._save_checkpoint(cur_epoch, is_best=False, is_last=True,
                                  best_agg_metric=best_agg_metric,
                                  best_epoch=best_epoch)
            if self.save_freq > 0 and (cur_epoch + 1) % self.save_freq == 0:
                self._save_checkpoint(cur_epoch, is_best=False,
                                      best_agg_metric=best_agg_metric,
                                      best_epoch=best_epoch)

        return best_agg_metric, best_epoch

    def train(self, wandb_run):
        start_time = time.time()
        best_agg_metric = (
            float("inf") if self.selection_mode == "min" else float("-inf")
        )
        best_epoch = 0

        self.log_config()

        if self.evaluate_only:
            self.validate(
                cur_epoch="provided",
                best_agg_metric=best_agg_metric,
                best_epoch=0,
                wandb_run=wandb_run,
            )
            return

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            self._load_checkpoint(self.resume_ckpt_path)
            if self.config.run_cfg.get("delete_resume_ckpt_after_load", False):
                self._delete_local_resume_checkpoint_after_load(self.resume_ckpt_path)
            # Restore best-tracking from the checkpoint so we don't overwrite
            # checkpoint_best.pth with a worse score on the first resumed epoch.
            resumed_metric = getattr(self, "_resumed_best_metric", None)
            resumed_epoch = getattr(self, "_resumed_best_epoch", None)
            if resumed_metric is not None:
                best_agg_metric = resumed_metric
            if resumed_epoch is not None:
                best_epoch = resumed_epoch

        for cur_epoch in range(self.start_epoch, self.max_epoch):
            if hasattr(self.train_loader, "set_epoch"):
                self.train_loader.set_epoch(cur_epoch)
            custom_epochs = self.datasets['mimic_cxr']['train'].custom_epochs_per_epoch if 'mimic_cxr' in self.datasets else self.datasets['train'].custom_epochs_per_epoch
            stop_training = False
            for custom_epoch in range(custom_epochs):
                if 'mimic_cxr' in self.datasets: #before first epoch
                    self.datasets['mimic_cxr']['train'].set_custom_epoch(custom_epoch)
                else:
                    self.datasets['train'].set_custom_epoch(custom_epoch) #resorted after creating dataloader
                # training phase
                if not self.evaluate_only:
                    logging.info("Start training")
                    train_stats = self.train_epoch(cur_epoch)
                    self.log_stats(split_name="train", stats=train_stats)

                # evaluation phase
                best_agg_metric, best_epoch = self.validate(cur_epoch, best_agg_metric, best_epoch, wandb_run)

                if (
                    not self.evaluate_only
                    and self.early_stop_patience > 0
                    and len(self.valid_splits) > 0
                    and cur_epoch - best_epoch >= self.early_stop_patience
                ):
                    logging.info(
                        "Early stopping at epoch %d: best val %s %.6f at epoch %d, "
                        "patience=%d, min_delta=%g.",
                        cur_epoch,
                        self.selection_metric,
                        best_agg_metric,
                        best_epoch,
                        self.early_stop_patience,
                        self.early_stop_min_delta,
                    )
                    try:
                        wandb_run.log(
                            {
                                "early_stop/epoch": cur_epoch,
                                "early_stop/best_epoch": best_epoch,
                                f"early_stop/best_val_{self.selection_metric}": best_agg_metric,
                            }
                        )
                    except Exception:
                        pass
                    stop_training = True

                if self.evaluate_only:
                    break

                if stop_training:
                    break

            if stop_training:
                break

            #dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

        # Run the held-out test split exactly once using the selected validation
        # checkpoint.  This avoids tuning epochs, thresholds, or hyperparameters
        # on test performance.
        if (
            not self.evaluate_only
            and self.test_splits
            and bool(self.config.run_cfg.get("finalize_test", False))
        ):
            best_path = self.output_dir / "checkpoint_best.pth"
            if best_path.exists():
                for split_name in self.test_splits:
                    logging.info("Final evaluation on held-out %s split.", split_name)
                    _, _, _, test_stats = self.eval_epoch(
                        split_name=split_name, cur_epoch="best"
                    )
                    if test_stats is not None:
                        self.log_stats(test_stats, f"final_{split_name}")
            else:
                logging.warning(
                    "No checkpoint_best.pth exists; skipping final held-out test evaluation."
                )


    def evaluate(self, cur_epoch="best", skip_reload=False):
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:
                test_logs[split_name] = self.eval_epoch(
                    split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                )

            return test_logs

    def train_epoch(self, epoch):
        # train
        self.model.train()

        return self.task.train_epoch(
            epoch=epoch,
            model=self.model,
            data_loader=self.train_loader,
            optimizer=self.optimizer,
            scaler=self.scaler,
            lr_scheduler=self.lr_scheduler,
            cuda_enabled=self.cuda_enabled,
            log_freq=self.log_freq,
            accum_grad_iters=self.accum_grad_iters,
            max_grad_norm=self.max_grad_norm,
        )

    @torch.no_grad()
    def eval_epoch(self, split_name, cur_epoch, skip_reload=False):
        """
        Evaluate the model on a given split.

        Args:
            split_name (str): name of the split to evaluate on.
            cur_epoch (int): current epoch.
            skip_reload_best (bool): whether to skip reloading the best checkpoint.
                During training, we will reload the best checkpoint for validation.
                During testing, we will use provided weights and skip reloading the best checkpoint .
        """
        data_loader = self.dataloaders.get(split_name, None)
        assert data_loader, "data_loader for split {} is None.".format(split_name)

        model = self.unwrap_dist_model(self.model)
        if (not skip_reload and cur_epoch == "best"):
            model = self._reload_best_model(model)
        model.eval()

        self.task.before_evaluation(
            model=model,
            dataset=self.datasets[split_name],
        )

        results = self.task.evaluation(model, data_loader, cuda_enabled=self.cuda_enabled)

        if results is not None and self.config.run_cfg.task != "image_text_pretrain_eval":
            metrics, gts = data_loader.dataset.evaluator.evaluate(results)
            return metrics, results, gts, None

        else:
            return None, None, None, results # for pre-training instead of predicting text we compute the val loss


    def unwrap_dist_model(self, model):
        if self.use_distributed:
            return model.module
        else:
            return model

    def create_loaders(
        self,
        datasets,
        num_workers,
        batch_sizes,
        is_trains,
        collate_fns,
        dataset_ratios=None,
    ):
        """
        Create dataloaders for training and validation.
        """

        def _create_loader(dataset, num_workers, bsz, is_train, collate_fn):
            # create a single dataloader for each split
            if isinstance(dataset, ChainDataset):
                # wds.WebdDataset instance are chained together
                # webdataset.DataPipeline has its own sampler and collate_fn
                loader = iter(
                    DataLoader(
                        dataset,
                        batch_size=bsz,
                        num_workers=num_workers,
                        pin_memory=True,
                    )
                )
            else:
                # map-style dataset are concatenated together
                # setup distributed sampler
                if self.use_distributed:
                    sampler = (
                        DistributedSampler(
                            dataset,
                            shuffle=True,
                            num_replicas=get_world_size(),
                            rank=get_rank(),
                        )
                        if is_train
                        else DistributedEvalSampler(
                            dataset,
                            num_replicas=get_world_size(),
                            rank=get_rank(),
                        )
                    )
                    if not self.use_dist_eval_sampler:
                        # e.g. retrieval evaluation
                        sampler = sampler if is_train else None
                else:
                    sampler = None

                loader_kwargs = dict(
                    batch_size=bsz,
                    num_workers=num_workers,
                    pin_memory=True,
                    sampler=sampler,
                    shuffle=sampler is None and is_train,
                    collate_fn=collate_fn,
                    drop_last=True if is_train else False,
                )
                if num_workers > 0:
                    loader_kwargs["prefetch_factor"] = 2
                    # Bound per-batch worker fetch time so a stalled/hung image
                    # read raises RuntimeError instead of blocking next(loader)
                    # forever. Normal batches take ~1s, so this only trips on a
                    # real stall. Override with run.dataloader_timeout (0 disables).
                    dl_timeout = float(self.config.run_cfg.get("dataloader_timeout", 600))
                    if dl_timeout > 0:
                        loader_kwargs["timeout"] = dl_timeout

                loader = DataLoader(dataset, **loader_kwargs)
                #loader = PrefetchLoader(loader)

                if is_train:
                    loader = IterLoader(loader, use_distributed=self.use_distributed)

            return loader

        loaders = []

        for dataset, bsz, is_train, collate_fn in zip(
            datasets, batch_sizes, is_trains, collate_fns
        ):
            if isinstance(dataset, list) or isinstance(dataset, tuple):
                loader = MultiIterLoader(
                    loaders=[
                        _create_loader(d, num_workers, bsz, is_train, collate_fn[i])
                        for i, d in enumerate(dataset)
                    ],
                    ratios=dataset_ratios,
                )
            else:
                loader = _create_loader(dataset, num_workers, bsz, is_train, collate_fn)

            loaders.append(loader)

        return loaders

    def _save_checkpoint(self, cur_epoch, is_best=False, is_last=False,
                         best_agg_metric=None, best_epoch=None):
        """
        Save the checkpoint at the current epoch.
        """
        model_no_ddp = self.unwrap_dist_model(self.model)
        param_grad_dic = {
            k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
        }
        state_dict = model_no_ddp.state_dict()
        for k in list(state_dict.keys()):
            if k in param_grad_dic.keys() and not param_grad_dic[k]:
                # delete parameters that do not require gradient
                del state_dict[k]
        include_training_state = not is_best
        optimizer_state = self.optimizer.state_dict() if include_training_state else None
        rng_state = None
        if include_training_state:
            local_rng = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }
            if dist.is_available() and dist.is_initialized():
                rng_state = [None] * get_world_size()
                dist.all_gather_object(rng_state, local_rng)
            else:
                rng_state = [local_rng]
        if not is_main_process():
            return
        save_to = os.path.join(
            self.output_dir,
            "checkpoint_{}.pth".format("best" if is_best else ("last" if is_last else cur_epoch)),
        )

        # Refuse to overwrite a prior good checkpoint with a poisoned one.
        # If model or optimizer state contains NaN/Inf, log and bail out
        # silently so training can keep trying to recover.
        model_bad, model_bad_count = _state_dict_has_non_finite(state_dict)
        opt_bad, opt_bad_count = (
            _state_dict_has_non_finite(optimizer_state)
            if include_training_state
            else (False, 0)
        )
        if model_bad or opt_bad:
            logging.error(
                f"Refusing to save checkpoint to {save_to}: "
                f"model has {model_bad_count} non-finite tensor(s), "
                f"optimizer has {opt_bad_count}. Prior checkpoint preserved."
            )
            return

        save_obj = {
            "model": state_dict,
            "config": self.config.to_dict(),
            "identity": self._checkpoint_identity(),
            # Provenance only: records which commit produced the weights, but is
            # NOT part of the resume identity check, so a plumbing-only commit
            # does not strand the checkpoint.
            "source_commit": self.config.run_cfg.get("source_commit"),
            "epoch": cur_epoch,
            "best_agg_metric": best_agg_metric,
            "best_epoch": best_epoch,
        }
        if include_training_state:
            save_obj["optimizer"] = optimizer_state
            save_obj["scaler"] = self.scaler.state_dict() if self.scaler else None
            save_obj["rng_by_rank"] = rng_state
        logging.info("Saving checkpoint at epoch {} to {}.".format(cur_epoch, save_to))
        torch.save(save_obj, save_to)

    def _reload_best_model(self, model):
        """
        Load the best checkpoint for evaluation.
        """
        checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")

        logging.info("Loading checkpoint from {}.".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        identity = checkpoint.get("identity")
        assert_checkpoint_identity(
            identity, self._checkpoint_identity(), "Best"
        )
        self._load_trainable_state(model, checkpoint["model"], checkpoint_path)
        return model

    def _load_checkpoint(self, url_or_filename):
        """
        Resume from a checkpoint.
        """
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device, weights_only=False)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device, weights_only=False)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        expected_identity = self._checkpoint_identity()
        assert_checkpoint_identity(
            checkpoint.get("identity"), expected_identity, "Resume"
        )
        model_bad, model_bad_count = _state_dict_has_non_finite(state_dict)
        if model_bad:
            logging.warning(
                f"Model state_dict contains {model_bad_count} non-finite tensor(s); "
                "loading anyway but training may need to recover via the NaN-loss guard."
            )
        model = self.unwrap_dist_model(self.model)
        self._load_trainable_state(model, state_dict, url_or_filename)
        
        finetune_classifier = self.config.run_cfg.get("finetune_classifier", False)
        if finetune_classifier:
            print("Unfreezing classifier parameters.")
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze MHCAC module
            for param in self.model.mhcac.parameters():
                param.requires_grad = True

            # The multi-view fusion modules are new trainable params and would
            # otherwise stay frozen by the blanket freeze above.
            view_fusion = getattr(self.unwrap_dist_model(self.model), "view_fusion", None)
            if view_fusion is not None:
                print("Unfreezing view_fusion parameters.")
                for param in view_fusion.parameters():
                    param.requires_grad = True

            # for param in self.model.image_embed_proj.parameters():
            #     param.requires_grad = True
            
            # for param in self.model.text_cls_proj.parameters():
            #     param.requires_grad = True
            
            # for param in self.model.image_embed_proj_norm.parameters():
            #     param.requires_grad = True
            
            # for param in self.model.text_cls_proj_norm.parameters():
            #     param.requires_grad = True
            # # Unfreeze QueryAggregator module
            # for param in self.model.aggregator.parameters():
            #     param.requires_grad = True

        # Initialize the optimizer using the property
        optimizer = self.optimizer
        
        print(summary(self.model, input_size=None, device='cpu'))

        # Skip loading optimizer state if it contains NaN/Inf (poisoned Adam
        # moments would corrupt model weights on the first update). Letting
        # Adam re-initialize is safe; checkpoint_best.pth intentionally omits
        # optimizer/scaler state to keep Kaggle disk usage low.
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is None:
            logging.warning("Optimizer state missing from checkpoint; optimizer will be re-initialized.")
        else:
            opt_bad, opt_bad_count = _state_dict_has_non_finite(optimizer_state)
            if opt_bad:
                logging.warning(
                    f"Optimizer state contains {opt_bad_count} non-finite tensor(s) — "
                    "skipping load. Optimizer will be re-initialized."
                )
            else:
                try:
                    optimizer.load_state_dict(optimizer_state)
                except ValueError as exc:
                    raise RuntimeError(
                        "Optimizer state is incompatible with this exact run"
                    ) from exc

        if self.scaler and "scaler" in checkpoint and checkpoint["scaler"] is not None:
            scaler_bad, scaler_bad_count = _state_dict_has_non_finite(checkpoint["scaler"])
            scaler_dead, scaler_scale = _scaler_state_is_degenerate(checkpoint["scaler"])
            if scaler_bad:
                logging.warning(
                    f"Scaler state contains {scaler_bad_count} non-finite tensor(s) — "
                    "skipping load. GradScaler will start fresh."
                )
            elif scaler_dead:
                logging.warning(
                    f"Scaler state has degenerate scale={scaler_scale} — "
                    "skipping load. GradScaler will start fresh."
                )
            else:
                self.scaler.load_state_dict(checkpoint["scaler"])

        rng_by_rank = checkpoint.get("rng_by_rank")
        rank = get_rank()
        if not isinstance(rng_by_rank, list) or rank >= len(rng_by_rank):
            raise RuntimeError("Resume checkpoint is missing rank-specific RNG state")
        rng = rng_by_rank[rank]
        # RNG restore is best-effort. Resume identity gates on dataset + config,
        # not RNG-level determinism (cudnn.benchmark is on, deterministic off),
        # so a state that will not round-trip on this torch build must not abort
        # the resume -- warn and continue with fresh RNG, like the scaler path
        # above. torch.set_rng_state needs a CPU uint8 ByteTensor.
        try:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch_rng = rng["torch"]
            if torch.is_tensor(torch_rng):
                torch_rng = torch_rng.cpu().to(torch.uint8)
            torch.set_rng_state(torch_rng)
            if torch.cuda.is_available() and rng.get("cuda"):
                torch.cuda.set_rng_state_all(rng["cuda"])
        except (TypeError, RuntimeError, ValueError, KeyError) as exc:
            logging.warning(
                "Could not restore saved RNG state (%s: %s); continuing with "
                "fresh RNG. Resume identity does not depend on it.",
                type(exc).__name__, exc,
            )

        self.start_epoch = checkpoint["epoch"] + 1
        # Restore best-tracking so resumed runs don't overwrite checkpoint_best.pth
        # with a worse score (old checkpoints without these keys fall back to None).
        self._resumed_best_metric = checkpoint.get("best_agg_metric", None)
        self._resumed_best_epoch = checkpoint.get("best_epoch", None)
        logging.info(
            "Resume checkpoint from {} (best_agg_metric={}, best_epoch={})".format(
                url_or_filename, self._resumed_best_metric, self._resumed_best_epoch
            )
        )

    def _delete_local_resume_checkpoint_after_load(self, checkpoint_path):
        if not checkpoint_path or is_url(checkpoint_path) or not os.path.isfile(checkpoint_path):
            return

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        if is_main_process():
            try:
                size_mb = os.path.getsize(checkpoint_path) / (1024 ** 2)
                os.remove(checkpoint_path)
                logging.info(
                    "Deleted local resume checkpoint after load: %s (%.1f MB freed)",
                    checkpoint_path,
                    size_mb,
                )

                parent = Path(checkpoint_path).parent
                if parent.exists():
                    try:
                        next(parent.iterdir())
                    except StopIteration:
                        parent.rmdir()
            except OSError as exc:
                logging.warning("Could not delete local resume checkpoint %s: %s", checkpoint_path, exc)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    @main_process
    def log_stats(self, stats, split_name, commit=True):
        if isinstance(stats, dict):
            log_stats = {**{f"{split_name}_{k}": v for k, v in stats.items()}}
            with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
            if wandb is not None:
                wandb_stats = {k: float(v) for k, v in log_stats.items()}
                wandb.log(wandb_stats, commit=commit)

        elif isinstance(stats, list):
            pass

    @main_process
    def log_config(self):
        with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(self.config.to_dict(), indent=4) + "\n")
