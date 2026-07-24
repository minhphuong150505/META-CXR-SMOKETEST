"""Pure, dependency-light helpers for checkpoint resume state.

Deliberately free of the heavy training imports (torchinfo / wandb / the LAVIS
model stack) so the resume logic is unit-testable on a CPU box — see
``tests/test_resume_state.py``.  ``runner_base.py`` wires these into the real
save/load path.

Three problems these functions exist to solve, all observed on the 2×T4 DDP
smoke run:

1. RNG restore raised ``TypeError: RNG state must be a torch.ByteTensor``.
   Root cause: the checkpoint was loaded with ``map_location=<cuda>``, which
   moves the saved RNG ByteTensors onto the GPU.  Both ``torch.set_rng_state``
   and ``torch.cuda.set_rng_state_all`` reject anything that is not a *CPU*
   uint8 ByteTensor.  ``as_cpu_byte_tensor`` coerces the state back to that
   exact type so the restore actually succeeds instead of being swallowed and
   silently replaced by a fresh RNG.

2. The GradScaler could be persisted with ``scale == 0`` (AMP collapse after
   repeated fp16 overflow).  ``scaler_state_health`` classifies the state so
   the save path can refuse to persist a dead scaler and the load path can
   report a *partial* resume instead of pretending it restored one.

3. The checkpoint schema had no version and conflated "last completed epoch"
   with "next epoch to run".  ``CHECKPOINT_VERSION`` plus explicit
   ``next_epoch`` (written by the runner) make the resume point unambiguous
   while staying backward compatible with v1 checkpoints.
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch


# Bump whenever the on-disk checkpoint schema changes in a way the load path
# must branch on.
#   v1 (implicit, no key): {model, optimizer, scaler, rng_by_rank, epoch, ...}
#   v2: adds checkpoint_version, next_epoch, scaler_healthy,
#       optimizer_steps_per_epoch, global_optimizer_step.
CHECKPOINT_VERSION = 2

_RNG_STREAMS = ("python", "numpy", "torch", "cuda")


def as_cpu_byte_tensor(state) -> torch.Tensor:
    """Coerce a saved RNG state to a contiguous CPU uint8 (ByteTensor).

    ``torch.load(map_location=<cuda>)`` moves RNG tensors onto the GPU;
    ``torch.set_rng_state`` / ``torch.cuda.set_rng_state_all`` then reject them
    with ``RNG state must be a torch.ByteTensor`` because they require a CPU
    ByteTensor.  This makes the restore idempotent regardless of where the
    tensor currently lives.
    """
    if not torch.is_tensor(state):
        state = torch.as_tensor(state)
    return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def capture_rng_state() -> dict:
    """Snapshot every RNG stream a training step advances.

    ``torch``/``cuda`` entries are native CPU ByteTensors; ``python``/``numpy``
    are their libraries' opaque state objects.  Safe to ``all_gather_object``
    across ranks and ``torch.save``.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(rng: dict) -> dict:
    """Restore RNG streams captured by :func:`capture_rng_state`.

    Returns ``{stream_name: bool}`` recording which streams were actually
    restored, so the caller can log full vs partial reproducibility rather than
    silently continuing on a fresh RNG.

    Raises ``TypeError``/``KeyError`` only when the dict is structurally invalid
    (a real bug that must surface).  A stream that cannot be restored on *this*
    machine for a legitimate reason — e.g. the checkpoint carries CUDA states
    for a different GPU count — degrades that one stream to ``False`` without
    taking the others down with it.
    """
    if not isinstance(rng, dict):
        raise TypeError(f"RNG state must be a dict, got {type(rng).__name__}")
    for key in ("python", "numpy", "torch"):
        if key not in rng:
            raise KeyError(f"RNG state is missing required stream '{key}'")

    restored = {name: False for name in _RNG_STREAMS}

    random.setstate(rng["python"])
    restored["python"] = True

    np.random.set_state(rng["numpy"])
    restored["numpy"] = True

    torch.set_rng_state(as_cpu_byte_tensor(rng["torch"]))
    restored["torch"] = True

    cuda_states = rng.get("cuda") or []
    if not cuda_states:
        # Nothing to restore (saved on a CPU-only box); not a failure.
        restored["cuda"] = True
    elif torch.cuda.is_available():
        sanitized = [as_cpu_byte_tensor(s) for s in cuda_states]
        device_count = torch.cuda.device_count()
        if len(sanitized) == device_count:
            torch.cuda.set_rng_state_all(sanitized)
            restored["cuda"] = True
        else:
            # Device count changed since the checkpoint was written: restore
            # what maps 1:1 and report the stream as only partially restored.
            for idx in range(min(len(sanitized), device_count)):
                torch.cuda.set_rng_state(sanitized[idx], idx)
            restored["cuda"] = False
    else:
        # Saved CUDA RNG but resuming without CUDA — cannot restore it.
        restored["cuda"] = False

    return restored


def scaler_state_health(state) -> tuple[str, float | None]:
    """Classify a ``GradScaler.state_dict()`` for save/load decisions.

    Returns ``(status, scale)`` where status is one of:

      ``"missing"``     no usable scaler state present.
      ``"nonfinite"``   scale is NaN/Inf.
      ``"degenerate"``  scale <= 0.  AMP has collapsed: after ~165 consecutive
                        fp16 overflows the scale halves down to fp32 underflow
                        and hits 0, after which every step divides by ~0, is
                        skipped, and the optimizer never advances again.
      ``"healthy"``     scale > 0 and finite.
    """
    if not isinstance(state, dict) or state.get("scale") is None:
        return ("missing", None)
    scale = state.get("scale")
    try:
        scale_val = float(scale.item() if torch.is_tensor(scale) else scale)
    except (TypeError, ValueError):
        return ("missing", None)
    if not math.isfinite(scale_val):
        return ("nonfinite", scale_val)
    if scale_val <= 0:
        return ("degenerate", scale_val)
    return ("healthy", scale_val)


def resolve_next_epoch(checkpoint: dict) -> int:
    """The first epoch a resume must run, i.e. ``start_epoch``.

    Prefers the explicit v2 ``next_epoch`` field; falls back to the v1
    convention (``epoch`` stores the *last completed* epoch, so the next one is
    ``epoch + 1``).
    """
    if "next_epoch" in checkpoint and checkpoint["next_epoch"] is not None:
        return int(checkpoint["next_epoch"])
    return int(checkpoint["epoch"]) + 1
