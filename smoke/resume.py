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
#   v3: adds scheduler (stateless config snapshot), runtime_contract,
#       global_step/optimizer_step/micro_step, per-rank dataloader generator
#       state, and lr_by_group. Written whenever resume_mode is honoured.
CHECKPOINT_VERSION = 3

_RNG_STREAMS = ("python", "numpy", "torch", "cuda", "dataloader_generator")

# The two resume contracts.  ``strict`` refuses to continue unless every piece
# of training state came back verbatim; ``best_effort`` restores what it can and
# is required to label the result PARTIAL.
RESUME_MODES = ("strict", "best_effort")


def normalize_resume_mode(value) -> str:
    mode = str(value or "strict").strip().lower()
    if mode not in RESUME_MODES:
        raise ValueError(
            f"run.resume_mode must be one of {RESUME_MODES}, got {value!r}"
        )
    return mode


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


def capture_rng_state(generator: torch.Generator | None = None) -> dict:
    """Snapshot every RNG stream a training step advances.

    ``torch``/``cuda`` entries are native CPU ByteTensors; ``python``/``numpy``
    are their libraries' opaque state objects.  Safe to ``all_gather_object``
    across ranks and ``torch.save``.

    ``generator`` is the ``torch.Generator`` handed to the train ``DataLoader``.
    It is a *separate* RNG stream from the global one -- it drives shuffling and
    worker seeding -- so restoring the global torch RNG alone does not reproduce
    the batch order.  ``None`` is recorded as absent, not as a failure.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dataloader_generator": (
            generator.get_state() if generator is not None else None
        ),
    }


def restore_rng_state(rng: dict, generator: torch.Generator | None = None) -> dict:
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

    gen_state = rng.get("dataloader_generator")
    if generator is None:
        # No generator on this side: only a *saved* state is unrestorable.
        restored["dataloader_generator"] = gen_state is None
    elif gen_state is None:
        # A generator exists now but the checkpoint carries no state for it,
        # so the shuffle stream cannot be continued.
        restored["dataloader_generator"] = False
    else:
        generator.set_state(as_cpu_byte_tensor(gen_state))
        restored["dataloader_generator"] = True

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


# --------------------------------------------------------------------------- #
# AMP loss scale                                                               #
# --------------------------------------------------------------------------- #
# Torch's default ``growth_interval`` is 2000 *optimizer* steps, but a Stage-1
# epoch is ~657 of them (10,509 micro-batches / accum_grad_iters=16).  The scale
# can therefore only ever ratchet DOWN inside a run: every intermittent fp16
# overflow halves it and it never gets the 2000 clean steps needed to grow back.
# Run e123 walked 65536 -> 0 that way over two epochs, after which `scaler.step`
# is a permanent no-op -- the weights stopped moving and both epochs reported
# bit-identical validation metrics.  A growth interval well inside one epoch
# lets the scale recover between spikes instead.
AMP_INIT_SCALE = 2.0 ** 14
AMP_GROWTH_INTERVAL = 200
# Below 1.0 the "scale" shrinks gradients rather than protecting them from fp16
# underflow, so halving further buys nothing and only walks the value down into
# the subnormal range where it eventually becomes exactly 0.  Reset at this
# floor instead.
AMP_MIN_SCALE = 1.0


def new_grad_scaler():
    """A GradScaler with the project's scale policy (see the constants above)."""
    try:
        return torch.amp.GradScaler(
            "cuda", init_scale=AMP_INIT_SCALE, growth_interval=AMP_GROWTH_INTERVAL
        )
    except (AttributeError, TypeError):  # torch < 2.4
        return torch.cuda.amp.GradScaler(
            init_scale=AMP_INIT_SCALE, growth_interval=AMP_GROWTH_INTERVAL
        )


def reset_scaler_scale(scaler) -> float:
    """Lift a collapsed scaler back to ``AMP_INIT_SCALE`` in place.

    Returns the scale that was replaced.  Used instead of letting the scale keep
    halving toward 0, which freezes the optimizer for the rest of the run.
    """
    state = scaler.state_dict()
    previous = float(state.get("scale", 0.0))
    state["scale"] = float(AMP_INIT_SCALE)
    state["_growth_tracker"] = 0
    scaler.load_state_dict(state)
    return previous


# --------------------------------------------------------------------------- #
# Scheduler                                                                    #
# --------------------------------------------------------------------------- #
# The LAVIS schedulers in ``model/lavis/common/optims.py`` are *stateless*: LR is
# a pure function of (cur_epoch, cur_step, steps_per_epoch) and the constructor
# arguments below.  There is therefore no running counter to restore -- but the
# constructor arguments ARE load-bearing, and silently changing one (the classic
# case: ``scheduler_max_epoch: 10`` becoming ``max_epoch: 2`` on the resume
# command line) rewrites the whole cosine curve mid-run.  Snapshotting them makes
# that a hard failure instead of an invisible LR jump.
SCHEDULER_FIELDS = (
    "max_epoch",
    "min_lr",
    "init_lr",
    "warmup_steps",
    "warmup_start_lr",
    "decay_rate",
)


def scheduler_state(scheduler) -> dict:
    """Snapshot a LAVIS scheduler.

    ``class`` plus the constructor fields fully determine its LR curve.  Any
    attribute the concrete scheduler does not define is simply omitted.
    """
    state = {"class": type(scheduler).__name__}
    for field in SCHEDULER_FIELDS:
        if hasattr(scheduler, field):
            value = getattr(scheduler, field)
            state[field] = float(value) if isinstance(value, (int, float)) else value
    return state


def diff_states(saved, current) -> list[str]:
    """Field names whose values differ between two flat dicts.

    A missing key on either side counts as a difference; ``None`` for ``saved``
    means the checkpoint predates the field and every key is reported.
    """
    if not isinstance(saved, dict):
        return sorted(current) if isinstance(current, dict) else ["<absent>"]
    keys = set(saved) | set(current)
    return sorted(k for k in keys if saved.get(k) != current.get(k))


# --------------------------------------------------------------------------- #
# Runtime contract                                                             #
# --------------------------------------------------------------------------- #
# Everything that must be identical for a resumed run to follow the same
# trajectory as an uninterrupted one, but which is NOT already covered by
# ``smoke.identity`` (dataset manifest + config fingerprint).
#
# Deliberately excluded because they legitimately change between the two halves
# of the same run, and including them would make every resume mismatch:
#   run.max_epoch, run.resume_ckpt_path, run.wandb_run_id, run.run_role,
#   run.output_dir, run.run_name, run.delete_resume_ckpt_after_load.
RUNTIME_CONTRACT_FIELDS = (
    "world_size",
    "batch_size_train",
    "accum_grad_iters",
    "num_workers",
    "persistent_workers",
    "seed",
    "amp",
    "deterministic",
    "lr_sched",
    "scheduler_max_epoch",
    "init_lr",
    "init_lr_cls",
    "init_lr_q",
    "min_lr",
    "warmup_lr",
    "warmup_steps",
    "weight_decay",
    "beta2",
    "max_grad_norm",
)


def build_runtime_contract(run_cfg, world_size: int) -> dict:
    """The subset of run config a strict resume must find unchanged."""
    contract = {}
    for field in RUNTIME_CONTRACT_FIELDS:
        value = run_cfg.get(field, None)
        if isinstance(value, (int, float, bool)) or value is None:
            contract[field] = value
        else:
            contract[field] = str(value)
    # world_size is observed, not configured: torchrun --nproc_per_node decides
    # it, and the YAML copy can disagree with reality.
    contract["world_size"] = int(world_size)
    return contract


# --------------------------------------------------------------------------- #
# Resume reporting                                                             #
# --------------------------------------------------------------------------- #
class ResumeReport:
    """Accumulates what a resume did and did not restore.

    ``strict`` mode turns any entry in ``missing`` into a ``RuntimeError``;
    ``best_effort`` prints them under a PARTIAL RESUME banner.  Keeping the two
    modes on one object is what stops the loader from calling a resume "FULL"
    while the scaler was quietly reset.
    """

    def __init__(self, mode: str):
        self.mode = normalize_resume_mode(mode)
        self.fields: dict[str, str] = {}
        self.missing: list[str] = []

    def ok(self, name: str, detail: str = "loaded") -> None:
        self.fields[name] = detail

    def fail(self, name: str, detail: str) -> None:
        self.fields[name] = detail
        self.missing.append(f"{name}: {detail}")

    @property
    def strict(self) -> bool:
        return self.mode == "strict"

    @property
    def status(self) -> str:
        return "PARTIAL" if self.missing else "FULL"

    def raise_if_strict(self) -> None:
        if self.strict and self.missing:
            raise RuntimeError(
                "run.resume_mode=strict requires a fully restored training "
                "state, but the following could not be restored:\n  - "
                + "\n  - ".join(self.missing)
                + "\nThis checkpoint cannot reproduce an uninterrupted run. "
                "Either fix the cause, or set run.resume_mode=best_effort to "
                "continue from a knowingly PARTIAL resume."
            )

    def render(self) -> str:
        title = (
            "STRICT RESUME VERIFIED" if self.strict and not self.missing
            else f"{self.status} RESUME"
        )
        lines = [f"===== {title} ====="]
        lines += [f"{k}={v}" for k, v in self.fields.items()]
        if self.missing:
            lines.append("missing_or_reset_states:")
            lines += [f"- {item}" for item in self.missing]
        lines.append(f"resume_mode={self.mode}")
        lines.append(f"resume_status={self.status}")
        lines.append("=" * (len(title) + 12))
        return "\n".join(lines)


def resolve_next_epoch(checkpoint: dict) -> int:
    """The first epoch a resume must run, i.e. ``start_epoch``.

    Prefers the explicit v2 ``next_epoch`` field; falls back to the v1
    convention (``epoch`` stores the *last completed* epoch, so the next one is
    ``epoch + 1``).
    """
    if "next_epoch" in checkpoint and checkpoint["next_epoch"] is not None:
        return int(checkpoint["next_epoch"])
    return int(checkpoint["epoch"]) + 1
