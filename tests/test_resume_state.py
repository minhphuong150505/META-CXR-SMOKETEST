"""CPU tests for the checkpoint resume mechanism (smoke/resume.py + the runner
contract it implements).

Scope: everything that governs whether a resumed run is identical to an
uninterrupted one — RNG capture/restore (per rank, correct dtype/device), the
GradScaler health classification, the epoch schema (last-completed vs
next-to-run), and an end-to-end continuous-vs-resume equivalence check on a tiny
model.

Out of scope here (needs a GPU / the 2xT4 Kaggle run, per CLAUDE.md): the full
DDP train-vs-resume equivalence across the real BLIP2 model, and the
preflight-process isolation. The notebook enforces those at runtime (preflight
uses a separate output_dir + fingerprint; cell 11 asserts the saved epoch equals
max_epoch-1). The equivalence test below exercises the same RNG + optimizer +
stateless-scheduler + epoch-counter machinery that path relies on.
"""

import io
import random

import numpy as np
import pytest
import torch

from model.lavis.common.optims import LinearWarmupCosineLRScheduler
from smoke.resume import (
    CHECKPOINT_VERSION,
    as_cpu_byte_tensor,
    capture_rng_state,
    resolve_next_epoch,
    restore_rng_state,
    scaler_state_health,
)


# --------------------------------------------------------------------------- #
# RNG capture / restore                                                       #
# --------------------------------------------------------------------------- #
def test_rng_round_trip_reproduces_every_stream():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    for _ in range(3):  # advance off the seed point
        random.random()
        np.random.rand()
        torch.rand(2)

    state = capture_rng_state()
    a_py = random.random()
    a_np = float(np.random.rand())
    a_th = torch.rand(5)

    # Perturb every stream, then restore and re-draw.
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restored = restore_rng_state(state)

    assert restored == {"python": True, "numpy": True, "torch": True, "cuda": True}
    assert random.random() == a_py
    assert float(np.random.rand()) == a_np
    assert torch.equal(torch.rand(5), a_th)


def test_as_cpu_byte_tensor_forces_cpu_uint8():
    # torch.get_rng_state() is uint8; map_location=cuda would change its device
    # (unreproducible on a CPU box) but an int64 copy reproduces the "wrong type"
    # the coercion must fix.
    raw = torch.get_rng_state()
    mangled = raw.to(torch.int64)
    fixed = as_cpu_byte_tensor(mangled)
    assert fixed.dtype == torch.uint8
    assert fixed.device.type == "cpu"
    assert fixed.is_contiguous()
    assert torch.equal(fixed, raw)


def test_restore_coerces_wrong_dtype_state_instead_of_reseeding():
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    state = capture_rng_state()
    expected = torch.rand(4)

    # A raw non-ByteTensor is exactly what blew up before: torch.set_rng_state
    # rejects it with "RNG state must be a torch.ByteTensor".
    with pytest.raises((TypeError, RuntimeError)):
        torch.set_rng_state(state["torch"].to(torch.int64))

    # restore_rng_state coerces it and actually reproduces the stream.
    state["torch"] = state["torch"].to(torch.int64)
    restored = restore_rng_state(state)
    assert restored["torch"] is True
    assert torch.equal(torch.rand(4), expected)


def test_restore_uses_the_requested_rank_slice():
    random.seed(2)
    np.random.seed(2)
    torch.manual_seed(2)
    rng0 = capture_rng_state()
    draw_rank0 = torch.rand(3)

    torch.manual_seed(4242)
    random.seed(4242)
    np.random.seed(4242)
    rng1 = capture_rng_state()
    draw_rank1 = torch.rand(3)

    rng_by_rank = [rng0, rng1]
    assert not torch.equal(draw_rank0, draw_rank1)

    restore_rng_state(rng_by_rank[1])  # this rank restores ITS OWN slice
    assert torch.equal(torch.rand(3), draw_rank1)


def test_restore_raises_on_structurally_invalid_state():
    with pytest.raises(TypeError):
        restore_rng_state(["not", "a", "dict"])
    with pytest.raises(KeyError):
        restore_rng_state({"python": random.getstate()})  # missing numpy/torch


# --------------------------------------------------------------------------- #
# GradScaler health                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "state, expected",
    [
        ({"scale": 65536.0}, "healthy"),
        ({"scale": torch.tensor(2.0)}, "healthy"),
        ({"scale": 0.0}, "degenerate"),
        ({"scale": -1.0}, "degenerate"),
        ({"scale": float("nan")}, "nonfinite"),
        ({"scale": float("inf")}, "nonfinite"),
        ({}, "missing"),
        (None, "missing"),
    ],
)
def test_scaler_state_health_classification(state, expected):
    status, _ = scaler_state_health(state)
    assert status == expected


def test_real_gradscaler_state_is_healthy():
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    if not scaler.is_enabled():
        # A GradScaler auto-disables (empty state_dict) when CUDA is absent, as
        # on this CPU test box. The health check is exercised on real dicts by
        # the parametrized test above; here we only assert the enabled scaler.
        pytest.skip("GradScaler disabled without CUDA")
    status, scale = scaler_state_health(scaler.state_dict())
    assert status == "healthy"
    assert scale > 0


# --------------------------------------------------------------------------- #
# Epoch schema: last-completed vs next-to-run                                 #
# --------------------------------------------------------------------------- #
def test_resolve_next_epoch_epoch0_complete_resumes_at_epoch1():
    # The exact case from the bug report: a checkpoint at the END of epoch 0
    # must resume at epoch 1, never re-run epoch 0.
    assert resolve_next_epoch({"epoch": 0, "next_epoch": 1}) == 1
    # v1 checkpoints (no next_epoch) fall back to epoch + 1.
    assert resolve_next_epoch({"epoch": 0}) == 1
    assert resolve_next_epoch({"epoch": 3, "next_epoch": 4}) == 4


def test_resolve_next_epoch_trusts_explicit_next_epoch():
    # A mid-epoch checkpoint could legitimately ask to re-run the same epoch.
    assert resolve_next_epoch({"epoch": 2, "next_epoch": 2}) == 2


def test_checkpoint_version_is_v2_or_newer():
    assert isinstance(CHECKPOINT_VERSION, int)
    assert CHECKPOINT_VERSION >= 2


# --------------------------------------------------------------------------- #
# End-to-end: continuous run == train-0 / save / resume / train-1            #
# --------------------------------------------------------------------------- #
class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.drop = torch.nn.Dropout(0.5)  # consumes torch RNG each forward
        self.fc2 = torch.nn.Linear(8, 1)

    def forward(self, x):
        return self.fc2(self.drop(torch.relu(self.fc1(x))))


def _build():
    torch.manual_seed(123)
    net = _Net()
    opt = torch.optim.AdamW(
        [
            {"params": net.fc1.parameters(), "lr": 1e-3, "lr_scale": 1.0},
            {"params": net.fc2.parameters(), "lr": 2e-3, "lr_scale": 2.0},
        ],
        lr=1e-3,
    )
    sched = LinearWarmupCosineLRScheduler(
        optimizer=opt, max_epoch=2, min_lr=1e-5, init_lr=1e-3,
        warmup_steps=2, warmup_start_lr=1e-5,
    )
    return net, opt, sched


def _run_epoch(net, opt, sched, X, Y, cur_epoch, steps_per_epoch, batch=4):
    net.train()
    perm = torch.randperm(X.shape[0])  # RNG-dependent, like the shuffling sampler
    losses = []
    for step in range(steps_per_epoch):
        sched.step(cur_epoch, step, steps_per_epoch=steps_per_epoch)
        idx = perm[step * batch:(step + 1) * batch]
        opt.zero_grad()
        loss = ((net(X[idx]) - Y[idx]) ** 2).mean()  # dropout consumes RNG
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_resume_matches_continuous_training():
    torch.manual_seed(0)
    X = torch.randn(16, 4)
    Y = torch.randn(16, 1)
    STEPS = 4
    TRAIN_SEED = 777

    # --- Continuous 2-epoch run --------------------------------------------- #
    net, opt, sched = _build()
    torch.manual_seed(TRAIN_SEED)
    _run_epoch(net, opt, sched, X, Y, 0, STEPS)
    cont_epoch1 = _run_epoch(net, opt, sched, X, Y, 1, STEPS)
    w_continuous = net.fc2.weight.detach().clone()

    # --- Interrupted run: epoch 0, checkpoint, resume, epoch 1 -------------- #
    net2, opt2, sched2 = _build()
    torch.manual_seed(TRAIN_SEED)
    _run_epoch(net2, opt2, sched2, X, Y, 0, STEPS)

    buf = io.BytesIO()
    torch.save(
        {
            "model": net2.state_dict(),
            "optimizer": opt2.state_dict(),
            "epoch": 0,
            "next_epoch": 1,
            "rng_by_rank": [capture_rng_state()],
        },
        buf,
    )

    # Simulate a brand-new process: unrelated RNG, fresh objects.
    torch.manual_seed(4242)
    for _ in range(5):
        torch.rand(10)

    buf.seek(0)
    ckpt = torch.load(buf, map_location="cpu", weights_only=False)
    net3, opt3, sched3 = _build()
    net3.load_state_dict(ckpt["model"])
    opt3.load_state_dict(ckpt["optimizer"])
    start_epoch = resolve_next_epoch(ckpt)
    assert start_epoch == 1  # epoch 0 is NOT re-run
    assert all(restore_rng_state(ckpt["rng_by_rank"][0]).values())
    resume_epoch1 = _run_epoch(net3, opt3, sched3, X, Y, start_epoch, STEPS)
    w_resumed = net3.fc2.weight.detach().clone()

    # First post-resume loss and final weights match the continuous run.
    assert resume_epoch1[0] == pytest.approx(cont_epoch1[0], abs=1e-6)
    assert torch.allclose(w_continuous, w_resumed, atol=1e-6)


def test_skipping_rng_restore_diverges_from_continuous():
    """Negative control: without RNG restore the resumed run diverges — proof
    that RNG restore is load-bearing, not decorative."""
    torch.manual_seed(0)
    X = torch.randn(16, 4)
    Y = torch.randn(16, 1)
    STEPS = 4

    net, opt, sched = _build()
    torch.manual_seed(777)
    _run_epoch(net, opt, sched, X, Y, 0, STEPS)
    _run_epoch(net, opt, sched, X, Y, 1, STEPS)
    w_continuous = net.fc2.weight.detach().clone()

    net2, opt2, sched2 = _build()
    torch.manual_seed(777)
    _run_epoch(net2, opt2, sched2, X, Y, 0, STEPS)
    buf = io.BytesIO()
    torch.save({"model": net2.state_dict(), "optimizer": opt2.state_dict()}, buf)

    buf.seek(0)
    ckpt = torch.load(buf, map_location="cpu", weights_only=False)
    net3, opt3, sched3 = _build()
    net3.load_state_dict(ckpt["model"])
    opt3.load_state_dict(ckpt["optimizer"])
    torch.manual_seed(4242)  # fresh RNG, deliberately NOT restored
    _run_epoch(net3, opt3, sched3, X, Y, 1, STEPS)

    assert not torch.allclose(w_continuous, net3.fc2.weight, atol=1e-6)
