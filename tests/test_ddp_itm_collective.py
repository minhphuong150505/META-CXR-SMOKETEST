"""Distributed regression test for the ITM collective structure (Task C).

Reproduces the *class* of DDP deadlock that killed the Stage-1 smoke run: a
rank whose local micro-batch has no valid report taking a data-dependent early
return BEFORE a collective, while a peer rank runs the collective and blocks
forever.

`blip2_qformer.Blip2Qformer._image_text_matching` cannot be imported on a CPU
box (it pulls transformers/timm/the whole GPU stack). So this test rebuilds the
EXACT collective sequence of that method -- the global `valid_all.sum() < 2`
gate, the three all-gathers, and the dummy-row participation for a rank with no
local valid report -- inside a tiny DDP module, and runs it on two real gloo
processes. The collectives are the thing that deadlocks; the BERT numerics are
not, so mirroring the structure is a faithful test of the bug and its fix.

Two variants are run:
  * FIXED  (global gate, always run the 3 gathers): must complete all 6 cases.
  * BUGGY  (`if not valid_mask.any(): return` local gate): must DEADLOCK on
           Case 1, proving the test actually has teeth.

Launch (no GPU needed):
    /home/phuong/venv/bin/python tests/test_ddp_itm_collective.py

The process re-execs itself as two gloo workers; mp.spawn segfaults on the
torch 2.12 CPU build, so a plain subprocess launch is used instead.
"""
import datetime
import os
import subprocess
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

WORLD_SIZE = 2
B, P, D = 4, 5, 32  # batch per rank, tokens, dim (tiny; only structure matters)


# ---------------------------------------------------------------------------
# Collectives -- local copies so the test does not import the GPU-heavy model.
# ---------------------------------------------------------------------------
def concat_all_gather(tensor):
    gathered = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def gather_with_local_grad(tensor):
    """Mirror Blip2Qformer._gather_with_local_grad: keep this rank's grad."""
    gathered = concat_all_gather(tensor.detach())
    bs = tensor.shape[0]
    start = dist.get_rank() * bs
    return torch.cat([gathered[:start], tensor, gathered[start + bs:]], dim=0)


class ItmProbe(nn.Module):
    """Trainable stand-in that reproduces _image_text_matching's collectives.

    `qformer` stands in for the Q-Former cross-attention and `itm_head` is the
    real 2-way head. Both carry gradient, so backward drives the same DDP
    gradient all-reduce that the production model does -- the second half of the
    deadlock class (grad-readiness order) is therefore also exercised.
    """

    def __init__(self, buggy=False):
        super().__init__()
        self.qformer = nn.Linear(D, D)
        self.itm_head = nn.Linear(D, 2)
        self.buggy = buggy

    def forward(self, image_embeds, text_ids, text_atts, valid_mask, sim):
        zero = image_embeds.sum() * 0.0
        valid_all = concat_all_gather(valid_mask)  # runs on every rank (pre-gate)

        if self.buggy:
            # The ORIGINAL bug: a LOCAL gate before the three gathers below.
            if not valid_mask.any():
                return zero
        else:
            # The FIX: a GLOBAL gate -- identical decision on every rank.
            if valid_all.sum() < 2:
                return zero

        image_embeds_all = gather_with_local_grad(image_embeds)   # collective 1
        text_ids_all = concat_all_gather(text_ids)                # collective 2
        text_atts_all = concat_all_gather(text_atts)              # collective 3

        local_valid = bool(valid_mask.any())
        rank = dist.get_rank()
        positive_indices = rank * B + torch.arange(B)

        with torch.no_grad():
            w = F.softmax(sim, dim=1).clone()
            w[:, ~valid_all] = 0
            w[torch.arange(B), positive_indices] = 0
            w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-12)

        if local_valid:
            local_indices = valid_mask.nonzero(as_tuple=True)[0]
        else:
            local_indices = torch.zeros(1, dtype=torch.long)
        # multinomial must never see an all-zero row: valid_all.sum() >= 2
        # guarantees at least one non-self valid candidate per row.
        neg_idx = torch.stack([torch.multinomial(w[i], 1).squeeze(0)
                               for i in local_indices])
        assert bool(valid_all[neg_idx].all()), "sampled an invalid negative"
        assert bool((neg_idx != positive_indices[local_indices]).all()), \
            "sampled the positive as a negative"

        img = torch.cat([image_embeds[local_indices],
                         image_embeds_all[neg_idx],
                         image_embeds[local_indices]], dim=0)
        n = local_indices.numel()
        feats = self.qformer(img).mean(dim=1)
        logits = self.itm_head(feats)
        labels = torch.cat([torch.ones(n, dtype=torch.long),
                            torch.zeros(2 * n, dtype=torch.long)])
        loss = F.cross_entropy(logits, labels)
        if not local_valid:
            loss = loss * 0.0  # keep params connected, add no gradient
        return loss


# Rank -> boolean mask, one entry per named case (Section 6 of the spec).
CASES = {
    "1 r0-empty/r1-valid":  {0: [0, 0, 0, 0], 1: [1, 0, 1, 0]},
    "2 both-empty":         {0: [0, 0, 0, 0], 1: [0, 0, 0, 0]},
    "3 global-single":      {0: [0, 0, 0, 0], 1: [1, 0, 0, 0]},
    "4 both-valid":         {0: [1, 0, 1, 0], 1: [0, 1, 0, 1]},
    "5 r0-valid/r1-empty":  {0: [1, 0, 1, 0], 1: [0, 0, 0, 0]},
}


def run_case(model, ddp, opt, mask_list, gen):
    rank = dist.get_rank()
    # requires_grad mirrors production: image_embeds is the trainable shared
    # projector's output, so the gate-return `image_embeds.sum()*0.0` stays
    # graph-connected and backward is valid even when a rank returns early.
    image = torch.randn(B, P, D, generator=gen).requires_grad_(True)
    text_ids = torch.randint(0, 100, (B, 8), generator=gen)
    text_atts = torch.ones(B, 8, dtype=torch.long)
    sim = torch.randn(B, WORLD_SIZE * B, generator=gen)
    valid = torch.tensor(mask_list, dtype=torch.bool)
    opt.zero_grad(set_to_none=True)
    loss = ddp(image, text_ids, text_atts, valid, sim)
    assert torch.isfinite(loss).all(), f"rank{rank} non-finite loss"
    loss.backward()
    opt.step()
    return float(loss)


def worker():
    rank = int(os.environ["RANK"])
    buggy = os.environ.get("BUGGY", "0") == "1"
    # Long pg timeout so a genuine deadlock hangs (the parent kills it) rather
    # than converting to a fast error that could be mistaken for a clean pass.
    dist.init_process_group(
        "gloo", rank=rank, world_size=WORLD_SIZE,
        timeout=datetime.timedelta(seconds=300),
    )
    torch.manual_seed(0)
    model = ItmProbe(buggy=buggy)
    ddp = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    gen = torch.Generator().manual_seed(1234 + rank)  # divergent per rank

    # In buggy mode run ONLY Case 1 so the desync is a clean hang: rank 0 (empty
    # mask) returns early, skips the 3 gathers, and arrives at the barrier below,
    # while rank 1 blocks forever on the first gather -> the exact production
    # deadlock. Running further cases would instead cross-match mismatched
    # collectives and crash, which is harder to interpret.
    cases = {"1 r0-empty/r1-valid": CASES["1 r0-empty/r1-valid"]} if buggy else CASES
    for name, per_rank in cases.items():
        loss = run_case(model, ddp, opt, per_rank[rank], gen)
        print(f"[rank{rank}] CASE {name}: loss={loss:.4f}", flush=True)

    if not buggy:
        # Case 6: 100 iterations of random, per-rank-divergent masks.
        for it in range(100):
            mask = torch.randint(0, 2, (B,), generator=gen).tolist()
            run_case(model, ddp, opt, mask, gen)
        print(f"[rank{rank}] CASE 6 stress (100 iters): OK", flush=True)

    dist.barrier()
    print(f"[rank{rank}] ALL OK", flush=True)
    # The barrier above already proves both ranks finished every case without
    # deadlock. destroy_process_group()'s gloo destructor aborts (SIGABRT) at
    # interpreter teardown on the torch 2.12 CPU build, so force a clean exit
    # once the real work is done; a genuine deadlock never reaches this line and
    # the parent kills it on timeout instead.
    try:
        dist.destroy_process_group()
    except Exception:
        pass
    sys.stdout.flush()
    os._exit(0)


def _spawn(buggy, timeout):
    env = {**os.environ, "WORLD_SIZE": str(WORLD_SIZE),
           "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29517",
           "ROLE": "worker", "BUGGY": "1" if buggy else "0"}
    procs = [subprocess.Popen([sys.executable, __file__],
                              env={**env, "RANK": str(r)})
             for r in range(WORLD_SIZE)]
    try:
        codes = [p.wait(timeout=timeout) for p in procs]
    except subprocess.TimeoutExpired:
        for p in procs:
            p.kill()
        for p in procs:
            p.wait()
        return None  # timed out == deadlocked
    return codes


def main():
    print("=== FIXED variant: must complete all 6 cases on 2 gloo ranks ===")
    codes = _spawn(buggy=False, timeout=120)
    assert codes == [0, 0], f"FIXED variant failed/deadlocked (exit={codes})"
    print("FIXED: both ranks exited 0 -> no deadlock, finite loss.\n")

    print("=== BUGGY variant (local gate): must DEADLOCK on Case 1 ===")
    codes = _spawn(buggy=True, timeout=15)
    assert codes is None, (
        f"BUGGY variant did NOT deadlock (exit={codes}); the test has no teeth"
    )
    print("BUGGY: deadlocked as expected (parent killed it after 15s).")
    print("\nall DDP ITM collective tests passed")


if __name__ == "__main__":
    if os.environ.get("ROLE") == "worker":
        worker()
    else:
        main()
