"""8-way data-split emulation arithmetic (pure python; embedded verbatim into train_gpu.py).

On Trainium the scored run has world 8: prepare._document_batches("train", rank, 8, 128) hands rank r
every 8th row group of the train shards, make_stream_dataloader cuts rank r's documents into
FF_STREAM_MIX=8 BOS-joined streams (row i of a micro-batch comes from stream i % 8), and one optimizer
step at accumulation phase k takes k consecutive micro-batches of FF_MB=8 rows from EVERY rank, then
averages the gradient across ranks (AsyncGradSync / sync_gradients: ReduceOp.AVG of per-rank means of
k micro-batches = the mean over all 8k micro-batches).

One GPU (world 1) with the plain loader would read every row group in one stream: a different token
order, so a different run. train_gpu.py --data-world 8 instead builds shares = data_world / world_size
generators, generator s under the pseudo-rank (rank * shares + s, data_world) so it reads exactly the
row groups Trainium rank r*shares+s reads, and feeds the k * shares micro-batches of an optimizer step
share by share: micro j comes from share j // k, so share s delivers its k micro-batches in its own
order, the same 8k * 8 rows in the same per-stream order as the 8-rank step. The loss seed becomes
1 / (k * shares) (backward_scaled / accum_seed_tensors), so the accumulated gradient is the same mean
the all-reduce would have produced (fp32 summation order aside). Under torchrun with N GPUs the same
code gives shares = 8 / N; with 8 GPUs it is the plain per-rank behaviour.

ShareCursor is the index arithmetic alone (no torch) so it can be unit-tested here; the trainer's
loader wraps it around the real generators.

No `from __future__ import annotations` here on purpose: this file is embedded mid-module.
"""

from typing import Dict, List, Tuple


class ShareCursor:
    """Which pseudo-rank (share) the next fetched micro-batch comes from.

    The trainer fetches micro-batch j+1 right after micro-batch j's backward, so the LAST fetch of a
    step is micro 0 of the NEXT step, made before the loop knows the next step's k. Micro 0 always maps
    to share 0 whatever k is, and the loop calls set_k(step_k) at the top of every step before any
    further fetch, so the cursor is always right: share = j // k, wrapping after k * shares fetches."""

    def __init__(self, shares: int, k: int):
        if shares < 1 or k < 1:
            raise ValueError("ShareCursor needs shares >= 1 and k >= 1, got %r, %r" % (shares, k))
        self.shares = int(shares)
        self.k = int(k)
        self.j = 0

    def set_k(self, k: int) -> None:
        """Called at the top of every optimizer step with that step's micro-batches per rank."""
        if k < 1:
            raise ValueError("k must be >= 1")
        if self.j not in (0, 1):
            raise RuntimeError("ShareCursor.set_k mid-step (j=%d): the loop must set k before its first in-step "
                               "fetch, right after the prefetched micro 0" % self.j)
        self.k = int(k)
        # the prefetched micro 0 was counted under the old k; when the new step has a single micro-batch
        # (k * shares == 1) that prefetch was the whole step, so the cursor is already at the next step's start
        self.j %= self.k * self.shares

    def next_share(self) -> int:
        share = self.j // self.k
        if share >= self.shares:
            raise RuntimeError("ShareCursor: j=%d with k=%d exceeds %d shares" % (self.j, self.k, self.shares))
        self.j += 1
        if self.j >= self.k * self.shares:
            self.j = 0
        return share

    @property
    def micro_per_step(self) -> int:
        return self.k * self.shares


def emulated_fetch_order(k_by_step: List[int], shares: int) -> List[List[Tuple[int, int]]]:
    """The trainer's fetch pattern as (share, share-local index) per micro of each step: the prime
    fetch before the loop, then after every micro's backward one more (the last one is the next step's
    micro 0). Returns, per step, the list of (share, m) that fed micro 0..k*shares-1 of that step."""
    cur = ShareCursor(shares, k_by_step[0] if k_by_step else 1)
    counters: Dict[int, int] = {}

    def fetch():
        s = cur.next_share()
        m = counters.get(s, 0)
        counters[s] = m + 1
        return s, m

    pending = fetch()                     # the prime: x, y = next(train_loader) before the loop
    out: List[List[Tuple[int, int]]] = []
    for k in k_by_step:
        cur.set_k(k)
        step_rows = [pending]
        for _ in range(k * shares - 1):
            step_rows.append(fetch())
        pending = fetch()                 # after the last micro's backward: next step's micro 0
        out.append(step_rows)
    return out


def trainium_step_rows(k_by_step: List[int], world: int) -> List[List[Tuple[int, int]]]:
    """Reference: the (rank, rank-local micro index) multiset an N-rank Trainium step consumes, rank
    by rank (rank r takes its k consecutive micro-batches from its own generator every step)."""
    counters = [0] * world
    out: List[List[Tuple[int, int]]] = []
    for k in k_by_step:
        rows = []
        for r in range(world):
            for _ in range(k):
                rows.append((r, counters[r]))
                counters[r] += 1
        out.append(rows)
    return out


def grad_scale_matches(k: int, shares: int, world_gpus: int = 1) -> bool:
    """1/(k*shares) seeds summed over k*shares micro-batches per GPU, then AVG over world_gpus GPUs,
    equals the Trainium mean over k * shares * world_gpus micro-batches (= 8k with data world 8)."""
    per_gpu = sum(1.0 / (k * shares) for _ in range(k * shares))          # = 1.0: the per-GPU mean
    avg = per_gpu / world_gpus * world_gpus                               # AVG all-reduce of equal means
    return abs(avg - 1.0) < 1e-12 and k * shares * world_gpus == k * (shares * world_gpus)
