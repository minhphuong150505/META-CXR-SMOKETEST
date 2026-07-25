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


class _CollapsingScaler:
    """A GradScaler whose every step overflows, so the scale only ever halves.

    This is what run e123 hit for real: with growth_interval (2000 optimizer
    steps) larger than an epoch (~657), intermittent fp16 overflows walk the
    scale monotonically down to 0, after which `step()` is a permanent no-op and
    the weights never move again.
    """

    def __init__(self, scale):
        self.scale_value = float(scale)
        self.skipped = 0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def get_scale(self):
        return self.scale_value

    def step(self, optimizer):
        self.skipped += 1  # always an overflow: the update is discarded

    def update(self):
        self.scale_value *= 0.5

    def state_dict(self):
        return {"scale": self.scale_value, "_growth_tracker": 0}

    def load_state_dict(self, state):
        self.scale_value = float(state["scale"])


def test_collapsing_amp_scale_is_reset_instead_of_freezing_training():
    from smoke.resume import AMP_INIT_SCALE

    model = _SquaredErrorModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scaler = _CollapsingScaler(scale=2.0)
    batches = [{"x": torch.tensor([1.0]), "y": torch.tensor([1.0])} for _ in range(4)]

    stats = BaseTask()._train_inner_loop(
        epoch=0,
        iters_per_epoch=len(batches),
        model=model,
        data_loader=batches,
        optimizer=optimizer,
        lr_scheduler=_NoOpScheduler(),
        scaler=scaler,
        cuda_enabled=False,
        accum_grad_iters=1,
        max_grad_norm=0,
        log_freq=100,
    )

    # 2.0 -> 1.0 -> 0.5 trips the floor and resets, then halves on iters 2 and 3.
    assert scaler.scale_value == pytest.approx(AMP_INIT_SCALE / 4)
    assert stats["amp_collapse_resets"] == 1
    assert stats["amp_overflow_steps"] == 4
    assert stats["optimizer_steps_taken"] == 0
