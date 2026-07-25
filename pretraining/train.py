"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

from omegaconf import OmegaConf
import model.lavis.tasks as tasks
from model.lavis.common.config import Config
from model.lavis.common.dist_utils import get_rank, is_main_process, init_distributed_mode
from model.lavis.common.logger import setup_logger

from local_config import WANDB_ENTITY, WANDB_PROJECT, VIS_ROOT
from model.lavis.common.registry import registry
from model.lavis.common.utils import now

# imports modules for registration
from model.lavis.common.optims import (
   LinearWarmupCosineLRScheduler,
   LinearWarmupStepLRScheduler,
)
from model.lavis.models.blip2_models.blip2_qformer import Blip2Qformer  # noqa: F401
from model.lavis.runners.runner_base import RunnerBase
from model.lavis.tasks.image_text_pretrain import ImageTextPretrainTask  # noqa: F401
from model.lavis.data.ReportDataset import MIMIC_CXR_Dataset


# python -m torch.distributed.run --standalone --nproc_per_node=2 -m pretraining.train --cfg-path pretraining/configs/mimic_cxr_2gpu.yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if config.run_cfg.get("deterministic", False):
        # Exact-resume verification mode. Every source of run-to-run kernel
        # nondeterminism is turned off so a resumed run can be compared bitwise
        # against a continuous one. This is materially slower -- it is for the
        # integration test and for debugging a resume, not for production
        # throughput.
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # CUBLAS needs a fixed workspace for deterministic GEMMs, and it reads
        # the variable when the CUDA context is created -- setting it here would
        # be too late, so require it from the launcher instead of pretending.
        if torch.cuda.is_available():
            if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
                raise RuntimeError(
                    "run.deterministic=true requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
                    "to be exported BEFORE the process starts (cuBLAS reads it at "
                    "CUDA context creation). Prefix your torchrun command with it."
                )
        if os.environ.get("PYTHONHASHSEED") is None:
            # Only a warning: PYTHONHASHSEED affects set/dict iteration order,
            # which this pipeline does not use to build batches. Recorded so the
            # report can say what was and was not pinned.
            logging.warning(
                "run.deterministic=true but PYTHONHASHSEED is unset; export "
                "PYTHONHASHSEED=%s before launching for full reproducibility.",
                config.run_cfg.seed,
            )
        # warn_only is NOT set: an op without a deterministic implementation must
        # surface by name so it can be fixed or explicitly documented, rather
        # than being silently downgraded to "probably fine".
        torch.use_deterministic_algorithms(True)
    else:
        # benchmark autotunes cuDNN conv kernels for the fixed 448x448 input -- a
        # real throughput win on the conv-heavy frozen encoders. It gives up exact
        # bit-reproducibility of the loss curve, which resume identity does not
        # depend on (that gates on dataset + config, not RNG-level determinism).
        cudnn.benchmark = True
        cudnn.deterministic = False


def get_runner_class(cfg):
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    return runner_cls


def main():
    registry.mapping['paths']['cache_root'] = '.'
    cfg = Config(parse_args())

    job_id = now()

    # NCCL watchdog: turn a stuck collective into a raised error with a traceback
    # (naming the rank + collective) instead of an indefinite silent hang, so any
    # residual data-dependent DDP desync is diagnosable rather than a 28-minute
    # deadlock. setdefault so an explicit env override still wins. Only the
    # TORCH_-prefixed name is used; the bare NCCL_ASYNC_ERROR_HANDLING was
    # deprecated in torch 2.2 and merely emits a warning on current torch.
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    # Initialize distributed training (reads RANK, WORLD_SIZE, LOCAL_RANK set by torchrun)
    init_distributed_mode(cfg)

    # Bridge cfg.gpu into OmegaConf so runner_base can read it via cfg.run_cfg.gpu
    if hasattr(cfg, 'gpu'):
        OmegaConf.update(cfg.config, "run.gpu", cfg.gpu)
    if hasattr(cfg, 'distributed'):
        OmegaConf.update(cfg.config, "run.distributed", cfg.distributed)

    setup_seeds(cfg)
    setup_logger()

    # Only rank 0 logs to wandb; other ranks use disabled mode to silence any stray calls
    if is_main_process():
        try:
            wandb_entity = cfg.run_cfg.get("wandb_entity", WANDB_ENTITY)
            wandb_run_id = cfg.run_cfg.get("wandb_run_id", None)
            wandb_resume = cfg.run_cfg.get("wandb_resume", None)
            wandb_kwargs = {
                "project": cfg.run_cfg.get("project_name", WANDB_PROJECT),
                "entity": wandb_entity if wandb_entity else None,
                "name": cfg.run_cfg.run_name,
            }
            if wandb_run_id:
                wandb_kwargs["id"] = wandb_run_id
                wandb_kwargs["resume"] = wandb_resume or "allow"
            wandb_run = wandb.init(
                **wandb_kwargs
            )
        except wandb.errors.UsageError:
            print("wandb: No API key found — logging disabled")
            wandb_run = wandb.init(mode="disabled")
    else:
        wandb_run = wandb.init(mode="disabled")

    cfg.pretty_print()

    task = tasks.setup_task(cfg)

    # Only MIMIC-CXR-JPG dataset
    datasets = {}
    datasets['mimic_cxr'] = {}
    truncate_train = cfg.run_cfg.get("truncate_train", None)
    truncate_val = cfg.run_cfg.get("truncate_val", None)
    truncate_test = cfg.run_cfg.get("truncate_test", None)

    if not cfg.run_cfg.evaluate:
        datasets['mimic_cxr']['train'] = MIMIC_CXR_Dataset(
            vis_processor=None, text_processor=None,
            vis_root=VIS_ROOT,
            split="train", cfg=cfg, truncate=truncate_train
        )
        datasets['mimic_cxr']['val'] = MIMIC_CXR_Dataset(
            vis_processor=None, text_processor=None,
            vis_root=VIS_ROOT,
            split="val", cfg=cfg, truncate=truncate_val
        )

        if len(cfg.run_cfg.get("test_splits", [])) > 0:
            datasets['mimic_cxr']['test'] = MIMIC_CXR_Dataset(
                vis_processor=None, text_processor=None,
                vis_root=VIS_ROOT,
                split="test", cfg=cfg, truncate=truncate_test
            )
    else:
        eval_splits = list(cfg.run_cfg.get("test_splits", []))
        if not eval_splits:
            eval_splits = list(cfg.run_cfg.get("valid_splits", []))
        if not eval_splits:
            raise ValueError("evaluate=true requires test_splits or valid_splits")
        for split in eval_splits:
            truncate = {
                "train": truncate_train,
                "val": truncate_val,
                "test": truncate_test,
            }.get(split)
            datasets['mimic_cxr'][split] = MIMIC_CXR_Dataset(
                vis_processor=None,
                text_processor=None,
                vis_root=VIS_ROOT,
                split=split,
                cfg=cfg,
                truncate=truncate,
            )

    model = task.build_model(cfg)

    # Fail fast, by name, if any tensor is still on the meta device: the
    # model-wide .to(device) would otherwise raise the opaque "Cannot copy out
    # of meta tensor". Names make the leaking submodule obvious.
    _meta = [
        n for n, t in
        (list(model.named_parameters()) + list(model.named_buffers()))
        if getattr(t, "is_meta", False)
    ]
    if _meta:
        raise RuntimeError(
            f"{len(_meta)} tensor(s) on the meta device after build_model: {_meta}"
        )

    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train(wandb_run)


if __name__ == "__main__":
    main()
