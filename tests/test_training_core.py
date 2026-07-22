import math

import pytest
import torch

from model.lavis.common.optims import LinearWarmupCosineLRScheduler
from model.lavis.tasks.base_task import BaseTask


class _NoOpScheduler:
    def step(self, cur_epoch, cur_step, steps_per_epoch=None):
        return None


class _SquaredErrorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, samples):
        prediction = self.weight * samples["x"]
        return {"loss": ((prediction - samples["y"]) ** 2).mean()}


def test_tail_accumulation_uses_actual_window_size():
    model = _SquaredErrorModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    batches = [
        {"x": torch.tensor([1.0]), "y": torch.tensor([1.0])},
        {"x": torch.tensor([1.0]), "y": torch.tensor([3.0])},
        {"x": torch.tensor([1.0]), "y": torch.tensor([5.0])},
    ]

    BaseTask()._train_inner_loop(
        epoch=0,
        iters_per_epoch=len(batches),
        model=model,
        data_loader=batches,
        optimizer=optimizer,
        lr_scheduler=_NoOpScheduler(),
        cuda_enabled=False,
        accum_grad_iters=2,
        max_grad_norm=0,
        log_freq=100,
    )

    # First update averages targets 1 and 3: w 0 -> 4. The one-item tail then
    # uses divisor 1: w 4 -> 6. Dividing the tail by 2 would incorrectly give 5.
    assert model.weight.item() == pytest.approx(6.0)


def test_scheduler_preserves_parameter_group_lr_ratio():
    p1 = torch.nn.Parameter(torch.tensor(0.0))
    p2 = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD(
        [
            {"params": [p1], "lr": 1.0e-4, "lr_scale": 1.0},
            {"params": [p2], "lr": 2.0e-4, "lr_scale": 2.0},
        ],
        lr=1.0e-4,
    )
    scheduler = LinearWarmupCosineLRScheduler(
        optimizer=optimizer,
        max_epoch=2,
        min_lr=1.0e-6,
        init_lr=1.0e-4,
        warmup_steps=2,
        warmup_start_lr=1.0e-6,
    )

    for epoch, update in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 4)]:
        scheduler.step(epoch, update, steps_per_epoch=5)
        assert math.isfinite(optimizer.param_groups[0]["lr"])
        assert optimizer.param_groups[1]["lr"] == pytest.approx(
            optimizer.param_groups[0]["lr"] * 2.0
        )
