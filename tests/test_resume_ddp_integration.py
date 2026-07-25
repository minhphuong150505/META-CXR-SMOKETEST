"""Integration test: a 2-rank DDP run interrupted at the end of epoch 0 and
resumed from ``checkpoint_last`` must reproduce the continuous run exactly.

    python -m pytest tests/test_resume_ddp_integration.py -v

What it really exercises
------------------------
The resume *state machinery* the runner depends on, wired together the same way
``runner_base``/``base_task`` wire it:

  * ``smoke.resume``: per-rank RNG capture/restore (python/numpy/torch +
    DataLoader generator), scaler health, ``resolve_next_epoch``,
    ``scheduler_state``/``diff_states``, ``ResumeReport``.
  * ``DistributedSampler`` + ``set_epoch(start_epoch)`` -- the batch order.
  * ``LinearWarmupCosineLRScheduler`` from ``model.lavis.common.optims``, driven
    per accumulation window exactly as ``_train_inner_loop`` drives it.
  * ``GradScaler`` unscale_ -> clip -> step -> update, and checkpointing strictly
    after ``update()``.
  * AdamW with per-group ``lr_scale``, gradient accumulation, ``drop_last``.

Deliberately NOT covered here, and not claimed to be: the real BLIP2 model, the
frozen vision encoders, CUDA/fp16 kernels and NCCL. Those need the 2xT4 box.
This runs on CPU with the gloo backend so it can gate every commit; a passing
run proves the plumbing is correct, not that the full model is bitwise
deterministic on a GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, DistributedSampler

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Same trick as tests/conftest.py, repeated here on purpose: mp.spawn starts a
# *fresh interpreter* that re-imports this module without ever loading conftest,
# so without this the child dies on `model/lavis/__init__.py` pulling in
# torchvision. Registering the packages as path-only lets Python resolve
# `model.lavis.common.optims` without executing that __init__.
for _name in ("model", "model.lavis"):
    if _name not in sys.modules:
        import types

        _pkg = types.ModuleType(_name)
        _pkg.__path__ = [str(_REPO_ROOT / Path(*_name.split(".")))]
        sys.modules[_name] = _pkg

from model.lavis.common.optims import LinearWarmupCosineLRScheduler  # noqa: E402
from smoke.resume import (  # noqa: E402
    ResumeReport,
    capture_rng_state,
    diff_states,
    resolve_next_epoch,
    restore_rng_state,
    scaler_state_health,
    scheduler_state,
)

# Mirrors the config block the task asks for.
SEED = 42
WORLD_SIZE = 2
BATCH_SIZE_TRAIN = 4
ACCUM_GRAD_ITERS = 2
NUM_WORKERS = 0
MAX_EPOCH = 2
SCHEDULER_MAX_EPOCH = 4  # deliberately != MAX_EPOCH: the scheduler_max_epoch trap
MICRO_BATCHES = 8  # per rank per epoch -> 4 optimizer steps
MAX_GRAD_NORM = 1.0


# --------------------------------------------------------------------------- #
# Tiny stand-in for the real pipeline                                          #
# --------------------------------------------------------------------------- #
class _StudyDataset(Dataset):
    """Carries a study id per row so batch *order* can be compared per rank."""

    def __init__(self, n=64):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, 8, generator=g)
        self.y = torch.randn(n, 1, generator=g)
        self.study_ids = [f"s{i:05d}" for i in range(n)]

    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "y": self.y[idx],
            "study_id": self.study_ids[idx],
            # Consumes the worker/global torch RNG, like the real augmentation
            # pipeline does -- so a mis-restored RNG shows up as a data change.
            "jitter": torch.randn(1),
        }


def _collate(batch):
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "jitter": torch.stack([b["jitter"] for b in batch]),
        "study_id": [b["study_id"] for b in batch],
    }


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)
        self.drop = torch.nn.Dropout(0.3)  # consumes torch RNG in forward
        self.fc2 = torch.nn.Linear(16, 1)
        # A non-persistent buffer, exactly like Blip2Qformer's ITC queue: absent
        # from state_dict(), so it must be checkpointed separately or the resumed
        # run starts with a different one.
        self.register_buffer("aux_queue", torch.zeros(4), persistent=False)

    def forward(self, x, jitter):
        out = self.fc2(self.drop(torch.relu(self.fc1(x))))
        # Make the loss depend on the queue so a lost queue changes the numbers.
        return out + self.aux_queue.mean() + 0.01 * jitter


def _hash_tensors(named):
    h = hashlib.sha256()
    for name, tensor in sorted(named):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _model_hash(model):
    return _hash_tensors(
        list(model.named_parameters()) + list(model.named_buffers())
    )


def _optimizer_hash(optimizer):
    # Walk param_groups in declaration order rather than hashing
    # `optimizer.state` directly: that dict is keyed by Parameter object, and its
    # iteration/`id()` order is not stable across processes, which would make the
    # hash differ between the continuous and the resumed run for no real reason.
    items = []
    for gi, group in enumerate(optimizer.param_groups):
        for pi, param in enumerate(group["params"]):
            state = optimizer.state.get(param, {})
            for key in ("exp_avg", "exp_avg_sq", "step"):
                if key in state:
                    value = state[key]
                    items.append(
                        (f"{gi}.{pi}.{key}", torch.as_tensor(value, dtype=torch.float64))
                    )
    return _hash_tensors(items)


def _rng_hash():
    state = capture_rng_state()
    h = hashlib.sha256()
    h.update(state["torch"].numpy().tobytes())
    h.update(repr(state["python"]).encode())
    h.update(repr(state["numpy"][1][:8]).encode())
    return h.hexdigest()


def _build(rank):
    torch.manual_seed(SEED)  # identical init on every rank, as DDP requires
    net = _Net()
    opt = torch.optim.AdamW(
        [
            {"name": "qformer", "params": list(net.fc1.parameters()),
             "lr": 1e-3, "lr_scale": 1.0, "weight_decay": 0.02},
            {"name": "classifier", "params": list(net.fc2.parameters()),
             "lr": 2e-3, "lr_scale": 2.0, "weight_decay": 0.0},
        ],
        lr=1e-3,
    )
    sched = LinearWarmupCosineLRScheduler(
        optimizer=opt, max_epoch=SCHEDULER_MAX_EPOCH, min_lr=1e-5,
        init_lr=1e-3, warmup_steps=3, warmup_start_lr=1e-5,
    )
    return net, opt, sched


def _make_loader(dataset, rank, generator):
    sampler = DistributedSampler(
        dataset, shuffle=True, num_replicas=WORLD_SIZE, rank=rank, seed=SEED
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE_TRAIN,
        sampler=sampler,
        num_workers=NUM_WORKERS,
        drop_last=True,
        collate_fn=_collate,
        generator=generator,
        persistent_workers=False,
    ), sampler


def _run_epoch(epoch, ddp_model, net, opt, sched, loader, sampler, scaler, probes):
    """One epoch, structured exactly like ``BaseTask._train_inner_loop``."""
    sampler.set_epoch(epoch)  # BEFORE iter(loader) — batch order depends on it
    it = iter(loader)
    updates_per_epoch = MICRO_BATCHES // ACCUM_GRAD_ITERS
    opt.zero_grad(set_to_none=True)

    epoch_probe = {
        "sampler_epoch": epoch,
        "model_hash_at_epoch_start": _model_hash(net),
        "optimizer_hash_at_epoch_start": _optimizer_hash(opt),
        "rng_hash_at_epoch_start": _rng_hash(),
        "scaler_scale_at_epoch_start": float(scaler.get_scale()),
        "lr_at_epoch_start": [g["lr"] for g in opt.param_groups],
    }
    first_batch_ids = None
    losses = []
    grad_norms = []

    for i in range(MICRO_BATCHES):
        batch = next(it)
        if i == 0:
            first_batch_ids = list(batch["study_id"])

        window_start = (i // ACCUM_GRAD_ITERS) * ACCUM_GRAD_ITERS
        window_size = min(ACCUM_GRAD_ITERS, MICRO_BATCHES - window_start)
        is_sync_step = i + 1 == window_start + window_size
        if i == window_start:
            sched.step(
                cur_epoch=epoch,
                cur_step=i // ACCUM_GRAD_ITERS,
                steps_per_epoch=updates_per_epoch,
            )

        ctx = torch.enable_grad() if is_sync_step else ddp_model.no_sync()
        with ctx:
            out = ddp_model(batch["x"], batch["jitter"])
            loss = ((out - batch["y"]) ** 2).mean()
            losses.append(float(loss.detach()))
            scaler.scale(loss / window_size).backward()

        if is_sync_step:
            scaler.unscale_(opt)
            gn = clip_grad_norm_(ddp_model.parameters(), MAX_GRAD_NORM)
            grad_norms.append(float(gn))
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            # Update the non-persistent buffer, like the ITC queue enqueue.
            # Blip2Qformer fills that queue from concat_all_gather, so every
            # rank holds an identical copy; all-reducing here reproduces that.
            # Feeding it the rank-LOCAL loss instead would make the replicas
            # diverge, and saving rank 0's copy would then be wrong.
            queue_input = torch.tensor([float(loss.detach())])
            dist.all_reduce(queue_input, op=dist.ReduceOp.SUM)
            net.aux_queue.mul_(0.9).add_(0.1 * float(queue_input) / WORLD_SIZE)
            if len(grad_norms) == 1:
                epoch_probe["model_hash_after_first_step"] = _model_hash(net)
                epoch_probe["optimizer_hash_after_first_step"] = _optimizer_hash(opt)
                epoch_probe["lr_after_first_step"] = [g["lr"] for g in opt.param_groups]

    epoch_probe.update(
        first_batch_study_ids=first_batch_ids,
        losses=losses,
        grad_norms=grad_norms,
        model_hash_at_epoch_end=_model_hash(net),
        scaler_scale_at_epoch_end=float(scaler.get_scale()),
    )
    probes[f"epoch{epoch}"] = epoch_probe
    return updates_per_epoch


def _worker(rank, mode, tmpdir):
    """mode: 'continuous' | 'interrupt' (runs epoch 0, saves) | 'resume'."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        import random

        import numpy as np

        seed = SEED + rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)

        dataset = _StudyDataset()
        net, opt, sched = _build(rank)
        ddp_model = DDP(net)
        scaler = torch.amp.GradScaler("cpu", enabled=True)
        generator = torch.Generator()
        generator.manual_seed(SEED + rank)

        ckpt_path = Path(tmpdir) / "checkpoint_last.pth"
        probes = {}
        start_epoch = 0
        optimizer_step = 0
        global_step = 0

        if mode == "resume":
            # ---- the load path, in the order runner_base does it ------------ #
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            report = ResumeReport("strict")

            net.load_state_dict(ckpt["model"], strict=True)
            report.ok("model", "strict-loaded")

            buffers = dict(net.named_buffers())
            for name, value in ckpt["nonpersistent_buffers"].items():
                buffers[name].copy_(value)
            report.ok("nonpersistent_buffers", "loaded")

            opt.load_state_dict(ckpt["optimizer"])
            report.ok("optimizer", "loaded")

            sched_diff = diff_states(ckpt["scheduler"], scheduler_state(sched))
            if sched_diff:
                report.fail("scheduler", f"changed: {sched_diff}")
            else:
                report.ok("scheduler", "loaded")

            status, scale = scaler_state_health(ckpt["scaler"])
            if status == "healthy" and ckpt.get("scaler_healthy", True):
                scaler.load_state_dict(ckpt["scaler"])
                report.ok("scaler", f"loaded scale={scale}")
            else:
                report.fail("scaler", f"{status} scale={scale}")

            # Each rank takes ITS OWN slice, and the generator is restored
            # BEFORE any iter(loader) is created.
            restored = restore_rng_state(
                ckpt["rng_by_rank"][rank], generator=generator
            )
            failed = [k for k, ok in restored.items() if not ok]
            if failed:
                report.fail(f"rng_rank_{rank}", "+".join(failed))
            else:
                report.ok(f"rng_rank_{rank}", "loaded")

            if diff_states(ckpt["lr_by_group"],
                           {str(g["name"]): float(g["lr"]) for g in opt.param_groups}):
                report.fail("lr_by_group", "lr changed across save/load")
            else:
                report.ok("lr_by_group", "matched")

            start_epoch = resolve_next_epoch(ckpt)
            optimizer_step = ckpt["optimizer_step"]
            global_step = ckpt["global_step"]
            assert ckpt["micro_step"] == 0
            report.raise_if_strict()  # must not raise
            probes["resume_banner"] = report.render()
            probes["resume_status"] = report.status

        loader, sampler = _make_loader(dataset, rank, generator)

        for epoch in range(start_epoch, MAX_EPOCH):
            updates = _run_epoch(
                epoch, ddp_model, net, opt, sched, loader, sampler, scaler, probes
            )
            optimizer_step += updates
            global_step += MICRO_BATCHES

            if mode == "interrupt" and epoch == 0:
                # ---- the save path: strictly AFTER scaler.update() and
                # optimizer.zero_grad(), i.e. at a clean optimizer boundary.
                local_rng = capture_rng_state(generator)
                rng_by_rank = [None] * WORLD_SIZE
                dist.all_gather_object(rng_by_rank, local_rng)
                status, _ = scaler_state_health(scaler.state_dict())
                assert status == "healthy", f"refusing to save a {status} scaler"
                if rank == 0:
                    torch.save(
                        {
                            "model": net.state_dict(),
                            "nonpersistent_buffers": {
                                n: b.detach().clone()
                                for n, b in net.named_buffers()
                                if n not in net.state_dict()
                            },
                            "optimizer": opt.state_dict(),
                            "scheduler": scheduler_state(sched),
                            "scaler": scaler.state_dict(),
                            "scaler_healthy": True,
                            "lr_by_group": {
                                str(g["name"]): float(g["lr"])
                                for g in opt.param_groups
                            },
                            "epoch": epoch,
                            "next_epoch": epoch + 1,
                            "global_step": global_step,
                            "optimizer_step": optimizer_step,
                            "micro_step": 0,
                            "rng_by_rank": rng_by_rank,
                        },
                        ckpt_path,
                    )
                dist.barrier()
                break

        probes["optimizer_step"] = optimizer_step
        probes["global_step"] = global_step
        probes["scheduler_state"] = scheduler_state(sched)
        (Path(tmpdir) / f"{mode}_rank{rank}.json").write_text(json.dumps(probes))
    finally:
        dist.destroy_process_group()


def _spawn(mode, tmpdir, port):
    os.environ["MASTER_PORT"] = str(port)
    mp.spawn(_worker, args=(mode, str(tmpdir)), nprocs=WORLD_SIZE, join=True)
    return [
        json.loads((Path(tmpdir) / f"{mode}_rank{r}.json").read_text())
        for r in range(WORLD_SIZE)
    ]


@pytest.mark.slow
def test_interrupted_ddp_run_matches_continuous_run(tmp_path):
    cont_dir = tmp_path / "continuous"
    resume_dir = tmp_path / "interrupted"
    cont_dir.mkdir()
    resume_dir.mkdir()

    # Run A: uninterrupted, epoch 0 -> epoch 1.
    continuous = _spawn("continuous", cont_dir, 29511)

    # Run B: epoch 0, checkpoint, process torn down, fresh processes, resume.
    _spawn("interrupt", resume_dir, 29512)
    assert (resume_dir / "checkpoint_last.pth").exists()
    resumed = _spawn("resume", resume_dir, 29513)

    for rank in range(WORLD_SIZE):
        a = continuous[rank]["epoch1"]
        b = resumed[rank]["epoch1"]

        assert resumed[rank]["resume_status"] == "FULL", resumed[rank]["resume_banner"]

        # 1-4: state at the START of the resumed epoch.
        assert b["sampler_epoch"] == 1, "sampler must not be rewound to epoch 0"
        assert b["model_hash_at_epoch_start"] == a["model_hash_at_epoch_start"]
        assert b["optimizer_hash_at_epoch_start"] == a["optimizer_hash_at_epoch_start"]
        assert b["rng_hash_at_epoch_start"] == a["rng_hash_at_epoch_start"]
        assert b["lr_at_epoch_start"] == a["lr_at_epoch_start"]
        assert b["scaler_scale_at_epoch_start"] == a["scaler_scale_at_epoch_start"]
        assert b["scaler_scale_at_epoch_start"] > 0

        # 5: first batch each rank sees must be identical, per rank.
        assert b["first_batch_study_ids"] == a["first_batch_study_ids"]

        # 6-8: the numbers the epoch produces.
        assert b["losses"] == a["losses"], "per-micro-batch losses diverged"
        assert b["grad_norms"] == a["grad_norms"]

        # 9-12: state after the first optimizer step, and at epoch end.
        assert b["model_hash_after_first_step"] == a["model_hash_after_first_step"]
        assert b["optimizer_hash_after_first_step"] == a["optimizer_hash_after_first_step"]
        assert b["lr_after_first_step"] == a["lr_after_first_step"]
        assert b["model_hash_at_epoch_end"] == a["model_hash_at_epoch_end"]

        # Counters and scheduler config.
        assert resumed[rank]["optimizer_step"] == continuous[rank]["optimizer_step"]
        assert resumed[rank]["global_step"] == continuous[rank]["global_step"]
        assert resumed[rank]["scheduler_state"] == continuous[rank]["scheduler_state"]

    # Both ranks must NOT have seen the same batch (proof the sampler shards).
    assert (
        continuous[0]["epoch1"]["first_batch_study_ids"]
        != continuous[1]["epoch1"]["first_batch_study_ids"]
    )
