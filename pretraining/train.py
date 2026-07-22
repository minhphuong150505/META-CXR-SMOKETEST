"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
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

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))
    return runner_cls


def main():
    registry.mapping['paths']['cache_root'] = '.'
    cfg = Config(parse_args())

    job_id = now()

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
    runner = RunnerBase(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )
    runner.train(wandb_run)


if __name__ == "__main__":
    main()
