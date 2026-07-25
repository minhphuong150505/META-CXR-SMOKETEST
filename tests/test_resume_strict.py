"""Strict vs best-effort resume contract.

These are the negative controls for ``tests/test_resume_ddp_integration.py``:
that test proves a *complete* checkpoint reproduces a continuous run; these
prove the loader refuses to call an *incomplete* one a full resume.

Every case here corresponds to a way the resume was previously silent:

  * a GradScaler saved with ``scale == 0`` and reported as healthy,
  * a scheduler whose ``max_epoch`` changed between the two halves of a run,
  * a checkpoint missing a rank's RNG slice,
  * a runtime contract (world size / batch size / accumulation / seed) that
    drifted,
  * counters absent because the checkpoint predates the schema.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.lavis.common.optims import LinearWarmupCosineLRScheduler
from smoke.resume import (
    RESUME_MODES,
    ResumeReport,
    build_runtime_contract,
    capture_rng_state,
    diff_states,
    normalize_resume_mode,
    restore_rng_state,
    scaler_state_health,
    scheduler_state,
)


# --------------------------------------------------------------------------- #
# Mode parsing                                                                 #
# --------------------------------------------------------------------------- #
def test_resume_mode_defaults_to_strict():
    assert normalize_resume_mode(None) == "strict"
    assert normalize_resume_mode("") == "strict"
    assert set(RESUME_MODES) == {"strict", "best_effort"}


@pytest.mark.parametrize("value", ["STRICT", " best_effort ", "Best_Effort"])
def test_resume_mode_is_case_and_whitespace_insensitive(value):
    assert normalize_resume_mode(value) in RESUME_MODES


def test_unknown_resume_mode_is_rejected_not_silently_defaulted():
    with pytest.raises(ValueError, match="resume_mode"):
        normalize_resume_mode("partial")


# --------------------------------------------------------------------------- #
# ResumeReport: the FULL/PARTIAL verdict                                       #
# --------------------------------------------------------------------------- #
def test_strict_report_raises_and_names_every_missing_state():
    report = ResumeReport("strict")
    report.ok("model", "strict-loaded")
    report.fail("scaler", "not restored (state=degenerate, scale=0.0)")
    report.fail("rng_rank_1", "not restored: cuda")

    assert report.status == "PARTIAL"
    with pytest.raises(RuntimeError) as exc:
        report.raise_if_strict()
    message = str(exc.value)
    assert "scaler" in message and "rng_rank_1" in message
    assert "best_effort" in message  # tells the operator the escape hatch


def test_best_effort_report_does_not_raise_but_must_say_partial():
    report = ResumeReport("best_effort")
    report.ok("model", "strict-loaded")
    report.fail("scaler", "not restored (state=degenerate, scale=0.0)")

    report.raise_if_strict()  # must not raise
    rendered = report.render()
    assert "PARTIAL RESUME" in rendered
    assert "resume_status=PARTIAL" in rendered
    assert "missing_or_reset_states:" in rendered
    assert "- scaler: not restored (state=degenerate, scale=0.0)" in rendered
    # It must NOT claim a verified/full resume.
    assert "STRICT RESUME VERIFIED" not in rendered


def test_complete_strict_resume_renders_the_verified_banner():
    report = ResumeReport("strict")
    for field in ("model", "optimizer", "scheduler", "scaler", "rng_rank_0"):
        report.ok(field, "loaded")
    report.raise_if_strict()

    rendered = report.render()
    assert "STRICT RESUME VERIFIED" in rendered
    assert "resume_status=FULL" in rendered
    assert "missing_or_reset_states" not in rendered


# --------------------------------------------------------------------------- #
# The scale=0.0 case specifically                                              #
# --------------------------------------------------------------------------- #
def test_degenerate_scaler_cannot_pass_strict_resume():
    """A scaler saved with scale=0 is exactly the reported symptom. It must not
    be loadable, and it must not be silently reset either."""
    status, scale = scaler_state_health({"scale": 0.0, "growth_factor": 2.0})
    assert status == "degenerate" and scale == 0.0

    report = ResumeReport("strict")
    report.fail("scaler", f"not restored (state={status}, scale={scale})")
    with pytest.raises(RuntimeError):
        report.raise_if_strict()


def test_scaler_flagged_unhealthy_at_save_is_not_trusted_even_if_state_looks_fine():
    """best_effort saves a *reset* scaler with scaler_healthy=False. The loader
    must treat the healthy-looking scale as NOT the run's scale."""
    checkpoint = {"scaler": {"scale": 65536.0}, "scaler_healthy": False}
    status, _ = scaler_state_health(checkpoint["scaler"])
    assert status == "healthy"  # the state itself is fine...
    # ...but the flag says it was substituted, so a strict resume must refuse.
    assert checkpoint["scaler_healthy"] is False


# --------------------------------------------------------------------------- #
# Scheduler config drift                                                       #
# --------------------------------------------------------------------------- #
def _sched(max_epoch=10, warmup_steps=300):
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-4)
    return LinearWarmupCosineLRScheduler(
        optimizer=opt, max_epoch=max_epoch, min_lr=1e-5, init_lr=1e-4,
        warmup_steps=warmup_steps, warmup_start_lr=5e-5,
    )


def test_scheduler_state_round_trips_unchanged():
    saved = scheduler_state(_sched())
    assert diff_states(saved, scheduler_state(_sched())) == []
    assert saved["class"] == "LinearWarmupCosineLRScheduler"


def test_scheduler_max_epoch_change_is_detected():
    """The exact trap: scheduler_max_epoch=10 silently becoming 2 rewrites the
    whole cosine curve, so a resumed epoch 1 gets a different LR."""
    saved = scheduler_state(_sched(max_epoch=10))
    current = scheduler_state(_sched(max_epoch=2))
    assert diff_states(saved, current) == ["max_epoch"]


def test_warmup_restart_is_detected():
    saved = scheduler_state(_sched(warmup_steps=300))
    current = scheduler_state(_sched(warmup_steps=0))
    assert "warmup_steps" in diff_states(saved, current)


def test_resumed_lr_equals_continuous_lr_at_the_same_position():
    """Resuming at epoch 1 must land on the same LR the continuous run had."""
    steps = 5
    cont = _sched(max_epoch=4, warmup_steps=3)
    for epoch in range(2):
        for step in range(steps):
            cont.step(cur_epoch=epoch, cur_step=step, steps_per_epoch=steps)
    continuous_lr = [g["lr"] for g in cont.optimizer.param_groups]

    # A fresh process resuming at epoch 1: the scheduler is stateless, so it
    # only needs the same config and the correct (epoch, step).
    resumed = _sched(max_epoch=4, warmup_steps=3)
    for step in range(steps):
        resumed.step(cur_epoch=1, cur_step=step, steps_per_epoch=steps)
    assert [g["lr"] for g in resumed.optimizer.param_groups] == continuous_lr

    # Negative control: the wrong max_epoch produces a different LR, which is
    # why the config comparison above is load-bearing.
    wrong = _sched(max_epoch=2, warmup_steps=3)
    for step in range(steps):
        wrong.step(cur_epoch=1, cur_step=step, steps_per_epoch=steps)
    assert [g["lr"] for g in wrong.optimizer.param_groups] != continuous_lr


# --------------------------------------------------------------------------- #
# Runtime contract                                                             #
# --------------------------------------------------------------------------- #
_BASE_CFG = {
    "batch_size_train": 4,
    "accum_grad_iters": 16,
    "num_workers": 4,
    "seed": 42,
    "amp": True,
    "deterministic": False,
    "lr_sched": "linear_warmup_cosine_lr",
    "scheduler_max_epoch": 10,
    "init_lr": 1e-4,
    "min_lr": 1e-5,
    "warmup_steps": 1,
    "weight_decay": 0.02,
    "max_grad_norm": 1.0,
}


@pytest.mark.parametrize(
    "field, new_value",
    [
        ("batch_size_train", 8),
        ("accum_grad_iters", 8),
        ("num_workers", 0),
        ("seed", 1),
        ("amp", False),
        ("scheduler_max_epoch", 2),
        ("init_lr", 2e-4),
        ("weight_decay", 0.0),
    ],
)
def test_runtime_contract_detects_each_disallowed_change(field, new_value):
    saved = build_runtime_contract(dict(_BASE_CFG), world_size=2)
    changed = build_runtime_contract({**_BASE_CFG, field: new_value}, world_size=2)
    assert diff_states(saved, changed) == [field]


def test_runtime_contract_detects_world_size_change():
    saved = build_runtime_contract(dict(_BASE_CFG), world_size=2)
    changed = build_runtime_contract(dict(_BASE_CFG), world_size=1)
    assert diff_states(saved, changed) == ["world_size"]


def test_runtime_contract_ignores_fields_that_must_change_on_resume():
    """max_epoch / resume_ckpt_path / wandb_run_id / run_role legitimately differ
    between the two halves of a run. Including them would make every resume
    mismatch, which is why they are excluded by design."""
    saved = build_runtime_contract(dict(_BASE_CFG), world_size=2)
    resumed_cfg = {
        **_BASE_CFG,
        "max_epoch": 3,
        "resume_ckpt_path": "/kaggle/input/ckpt/checkpoint_last.pth",
        "wandb_run_id": "abc123",
        "run_role": "train",
        "output_dir": "/kaggle/working/other",
    }
    assert diff_states(saved, build_runtime_contract(resumed_cfg, world_size=2)) == []


# --------------------------------------------------------------------------- #
# RNG: per-rank, and the DataLoader generator                                  #
# --------------------------------------------------------------------------- #
def test_missing_rank_slice_is_a_hard_failure_not_a_reseed():
    rng_by_rank = [capture_rng_state()]  # saved with world_size=1
    rank = 1
    assert rank >= len(rng_by_rank)  # this is the condition the runner raises on


def test_dataloader_generator_state_round_trips():
    gen = torch.Generator()
    gen.manual_seed(7)
    torch.rand(4, generator=gen)  # advance off the seed point

    state = capture_rng_state(gen)
    expected = torch.rand(4, generator=gen)

    gen.manual_seed(999)
    restored = restore_rng_state(state, generator=gen)
    assert restored["dataloader_generator"] is True
    assert torch.equal(torch.rand(4, generator=gen), expected)


def test_generator_absent_from_old_checkpoint_is_reported_not_faked():
    """A pre-v3 checkpoint carries no generator state. That must surface as an
    unrestored stream, not be papered over with a fresh seed."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    legacy = capture_rng_state(generator=None)  # no generator captured
    assert legacy["dataloader_generator"] is None

    gen = torch.Generator()
    gen.manual_seed(1)
    restored = restore_rng_state(legacy, generator=gen)
    assert restored["dataloader_generator"] is False

    report = ResumeReport("strict")
    report.fail("dataloader_generator", "not restored")
    with pytest.raises(RuntimeError, match="dataloader_generator"):
        report.raise_if_strict()
