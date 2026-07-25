"""Report what a Stage-1 checkpoint actually contains, and whether it can be
used for a *strict* (exact) resume.

    python scripts/inspect_checkpoint.py /path/to/checkpoint_last.pth

Prints one row per expected key -- exists / type / valid / summary -- then a
verdict. It never modifies the checkpoint: a scaler saved with ``scale=0`` is
reported as disqualifying rather than patched, because there is no value that
could be substituted for the collapsed scale that would reproduce the original
run's trajectory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smoke.resume import (  # noqa: E402
    CHECKPOINT_VERSION,
    resolve_next_epoch,
    scaler_state_health,
)

# Keys a v3 checkpoint written at an epoch boundary must carry for a strict
# resume. `required` rows that are absent or invalid disqualify the checkpoint.
REQUIRED = (
    "model",
    "optimizer",
    "scheduler",
    "scaler",
    "rng_by_rank",
    "epoch",
    "next_epoch",
    "global_step",
    "optimizer_step",
    "micro_step",
    "lr_by_group",
    "identity",
    "runtime_contract",
    "source_commit",
    "checkpoint_version",
)
OPTIONAL = ("best_agg_metric", "best_epoch", "scaler_healthy", "config")

# Extra model state that lives OUTSIDE model.state_dict() in some architectures.
# In this repo the ITC negative queue and its pointer are registered buffers on
# Blip2Qformer, so they travel inside "model" -- these names are checked there
# rather than as top-level keys, and duplicating them would be wrong.
IN_MODEL_STATE_PATTERNS = (
    "queue",
    "ptr",
    "teacher",
    "momentum",
    "ema",
    "running_",
    "view_fusion",
    "mhcac",
)


def _summarize(key, value):
    """(valid: bool | None, summary: str) for one checkpoint entry."""
    if key == "model":
        if not isinstance(value, dict):
            return False, f"expected dict, got {type(value).__name__}"
        n_bad = sum(
            1
            for v in value.values()
            if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()
        )
        return n_bad == 0, f"{len(value)} tensors, {n_bad} non-finite"

    if key == "optimizer":
        if not isinstance(value, dict):
            return False, f"expected dict, got {type(value).__name__}"
        groups = value.get("param_groups", [])
        state = value.get("state", {})
        with_moments = sum(1 for s in state.values() if "exp_avg" in s)
        lrs = [g.get("lr") for g in groups]
        return (
            with_moments > 0,
            f"{len(groups)} param groups, lr={lrs}, "
            f"{with_moments}/{len(state)} tensors carry Adam moments",
        )

    if key == "scheduler":
        if not isinstance(value, dict):
            return False, f"expected dict, got {type(value).__name__}"
        return True, ", ".join(f"{k}={v}" for k, v in sorted(value.items()))

    if key == "scaler":
        status, scale = scaler_state_health(value)
        return status == "healthy", f"status={status} scale={scale}"

    if key == "rng_by_rank":
        if not isinstance(value, list):
            return False, f"expected list, got {type(value).__name__}"
        rows = []
        for rank, entry in enumerate(value):
            if not isinstance(entry, dict):
                rows.append(f"rank{rank}=INVALID")
                continue
            streams = [
                name
                for name in ("python", "numpy", "torch", "cuda", "dataloader_generator")
                if entry.get(name) is not None
            ]
            rows.append(f"rank{rank}=[{'+'.join(streams)}]")
        complete = all(
            isinstance(e, dict) and all(e.get(s) is not None for s in ("python", "numpy", "torch"))
            for e in value
        )
        return complete, f"{len(value)} rank slice(s): " + " ".join(rows)

    if key == "identity":
        if not isinstance(value, dict):
            return False, f"expected dict, got {type(value).__name__}"
        return all(value.values()), ", ".join(
            f"{k}={str(v)[:16]}..." for k, v in sorted(value.items())
        )

    if key == "runtime_contract":
        if not isinstance(value, dict):
            return False, f"expected dict, got {type(value).__name__}"
        return True, ", ".join(f"{k}={v}" for k, v in sorted(value.items()))

    if key == "checkpoint_version":
        return int(value) >= CHECKPOINT_VERSION, f"v{value} (loader expects v{CHECKPOINT_VERSION})"

    if key == "config":
        return True, "<full run config>"

    return True, repr(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="path to checkpoint_last.pth")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise SystemExit(f"{args.checkpoint} is not a checkpoint dict")

    rows = []
    disqualifiers = []
    for key in REQUIRED + OPTIONAL:
        required = key in REQUIRED
        if key not in ckpt or ckpt[key] is None:
            rows.append((key, "no", "-", "-", "absent"))
            if required:
                disqualifiers.append(f"{key} is absent")
            continue
        value = ckpt[key]
        valid, summary = _summarize(key, value)
        rows.append(
            (key, "yes", type(value).__name__, {True: "yes", False: "NO"}[bool(valid)], summary)
        )
        if required and not valid:
            disqualifiers.append(f"{key} is present but invalid: {summary}")

    widths = [max(len(str(r[i])) for r in rows + [("key", "exists", "type", "valid", "summary")])
              for i in range(4)]
    header = ("key", "exists", "type", "valid", "summary")
    fmt = "  ".join(f"{{:<{w}}}" for w in widths) + "  {}"
    print(fmt.format(*header))
    print("-" * (sum(widths) + 8 + 40))
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))

    # Model-internal auxiliary state (queue / pointer / teacher / EMA / fusion).
    model_state = ckpt.get("model")
    if isinstance(model_state, dict):
        print("\nauxiliary model state carried inside model.state_dict():")
        found = False
        for pattern in IN_MODEL_STATE_PATTERNS:
            hits = [k for k in model_state if pattern in k]
            if hits:
                found = True
                print(f"  {pattern:<12} {len(hits):>4} tensor(s)  e.g. {hits[0]}")
        if not found:
            print("  (none matched — this model keeps no queue/EMA/teacher buffers)")

    if "epoch" in ckpt:
        print(
            f"\nepoch: last_completed={ckpt['epoch']} -> resume start_epoch="
            f"{resolve_next_epoch(ckpt)}"
        )

    print()
    if disqualifiers:
        print("VERDICT: this checkpoint is NOT eligible for strict/exact resume.")
        for item in disqualifiers:
            print(f"  - {item}")
        status, scale = scaler_state_health(ckpt.get("scaler"))
        if status in ("degenerate", "nonfinite"):
            print(
                "\n  The GradScaler scale cannot be repaired by hand. Substituting "
                "any value (65536 or otherwise) would put AMP on a different "
                "trajectory than the run this checkpoint came from, so the result "
                "would not be an exact resume — it would only look like one. "
                "Use run.resume_mode=best_effort and accept a PARTIAL resume, or "
                "restart the affected epoch."
            )
        return 1

    print("VERDICT: eligible for strict/exact resume.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
