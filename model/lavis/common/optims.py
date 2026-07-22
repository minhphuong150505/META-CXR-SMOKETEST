"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import math

from model.lavis.common.registry import registry

# Epoch during which linear warmup applies; every other epoch follows the
# cosine schedule. 0 = warmup at the start of training (upstream LAVIS
# behaviour). This was 5, which skipped warmup entirely at epoch 0 and instead
# dropped the LR back to warmup_lr in the middle of training.
current_epoch = 0


def _set_lr(optimizer, base_lr):
    """Set a base LR while preserving per-parameter-group LR ratios.

    Optimizer groups may carry ``lr_scale`` (for example, MHCAC versus
    Q-Former).  The old schedulers overwrote every group with the same scalar,
    silently discarding the configured group-specific learning rates.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = base_lr * float(param_group.get("lr_scale", 1.0))


@registry.register_lr_scheduler("linear_warmup_step_lr")
class LinearWarmupStepLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        decay_rate=1,
        warmup_start_lr=-1,
        warmup_steps=0,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.decay_rate = decay_rate

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step, steps_per_epoch=None):
        if cur_epoch == 0 and cur_step < self.warmup_steps:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            step_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
                decay_rate=self.decay_rate,
            )


@registry.register_lr_scheduler("linear_warmup_cosine_lr")
class LinearWarmupCosineLRScheduler:
    def __init__(
        self,
        optimizer,
        max_epoch,
        min_lr,
        init_lr,
        warmup_steps=0,
        warmup_start_lr=-1,
        **kwargs
    ):
        self.optimizer = optimizer

        self.max_epoch = max_epoch
        self.min_lr = min_lr

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr if warmup_start_lr >= 0 else init_lr

    def step(self, cur_epoch, cur_step, steps_per_epoch=None):
        # ``cur_step`` counts optimizer updates, not micro-batches.  When the
        # caller provides steps_per_epoch we can warm up once and decay smoothly
        # over all full-data updates, including when warmup spans an epoch.
        if steps_per_epoch is not None:
            global_step = cur_epoch * steps_per_epoch + cur_step
            total_steps = max(self.max_epoch * steps_per_epoch, 1)
            if global_step < self.warmup_steps:
                warmup_lr_schedule(
                    step=global_step,
                    optimizer=self.optimizer,
                    max_step=self.warmup_steps,
                    init_lr=self.warmup_start_lr,
                    max_lr=self.init_lr,
                )
            else:
                cosine_lr_schedule_steps(
                    optimizer=self.optimizer,
                    step=global_step - self.warmup_steps,
                    max_step=max(total_steps - self.warmup_steps, 1),
                    init_lr=self.init_lr,
                    min_lr=self.min_lr,
                )
        elif cur_epoch == current_epoch:
            warmup_lr_schedule(
                step=cur_step,
                optimizer=self.optimizer,
                max_step=self.warmup_steps,
                init_lr=self.warmup_start_lr,
                max_lr=self.init_lr,
            )
        else:
            cosine_lr_schedule(
                epoch=cur_epoch,
                optimizer=self.optimizer,
                max_epoch=self.max_epoch,
                init_lr=self.init_lr,
                min_lr=self.min_lr,
            )


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (
        1.0 + math.cos(math.pi * epoch / max_epoch)
    ) + min_lr
    _set_lr(optimizer, lr)


def cosine_lr_schedule_steps(optimizer, step, max_step, init_lr, min_lr):
    """Cosine decay indexed by optimizer updates."""
    progress = min(max(float(step) / max(float(max_step), 1.0), 0.0), 1.0)
    lr = (init_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)) + min_lr
    _set_lr(optimizer, lr)


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max(max_step, 1))
    _set_lr(optimizer, lr)


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    _set_lr(optimizer, lr)
