"""Both ranks must draw the same batch order, and accelerate must split it exactly once.

Simulates 2 ranks, each with its own GoldLengthGroupedBatchSampler (as separate processes
would have) and a differently-perturbed global `random` state (as Dataset.__getitem__
perturbs it during training). Wraps each in accelerate's real BatchSamplerShard and checks,
per epoch:
  - both ranks' samplers produce the identical batch list;
  - rank batches are disjoint and together cover every full batch exactly once;
  - both ranks get the same number of batches (DDP/FSDP collectives stay in lockstep);
  - every batch has a single gold count; epochs differ; a rerun reproduces the order.
A negative control reruns the split with the old behaviour (global, rank-divergent `random`)
to show the check detects duplicated/skipped examples.

Pure CPU, no model. Run from the repo root:
    python training/inf_retriever/tests/test_sampler_sharding.py \
        [--train_data data/training/filtered/qampari/train_data.jsonl]
Without --train_data it uses QAMPARI's gold-count distribution (5/6/7/8: 11785/7443/5138/4657).
"""
import argparse
import json
import os
import random
import sys

from accelerate.data_loader import BatchSamplerShard

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.finetuning_data import GoldLengthGroupedBatchSampler  # noqa: E402

BATCH, NPROC, EPOCHS, SEED = 50, 2, 3, 0


def load_gold_counts(path):
    if path is None:
        counts = {5: 11785, 6: 7443, 7: 5138, 8: 4657}
        gold = [k for k, n in counts.items() for _ in range(n)]
        random.Random(123).shuffle(gold)
        return gold
    with open(path) as f:
        return [len(json.loads(line).get("positive_ctxs", [])) for line in f]


def rank_epochs(gold, rank, perturb_seed, sampler_cls=GoldLengthGroupedBatchSampler, **kw):
    """One simulated rank: its own sampler + shard, with the global RNG perturbed per epoch."""
    sampler = sampler_cls(gold, BATCH, drop_last=True, shuffle=True, **kw)
    shard = BatchSamplerShard(sampler, num_processes=NPROC, process_index=rank, split_batches=False)
    perturb = random.Random(perturb_seed)
    out = []
    for _ in range(EPOCHS):
        random.seed(perturb.random())  # rank-specific global-RNG state, like __getitem__ leaves it
        out.append(list(shard))
    return out


class OldSampler(GoldLengthGroupedBatchSampler):
    """Pre-fix behaviour: shuffles with the global `random` (diverges across ranks)."""

    def __iter__(self):
        all_batches = []
        for _c, idx in sorted(self.groups.items()):
            idx = idx.copy()
            random.shuffle(idx)
            all_batches += [idx[s:s + self.batch_size] for s in range(0, len(idx), self.batch_size)
                            if not (self.drop_last and len(idx[s:s + self.batch_size]) < self.batch_size)]
        random.shuffle(all_batches)
        yield from all_batches


def coverage(r0, r1, n):
    seen = [i for b in r0 + r1 for i in b]
    return len(seen), len(set(seen)), n - len(set(seen))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_data", default=None)
    args = ap.parse_args()
    gold = load_gold_counts(args.train_data)
    n = len(gold)

    # Full per-epoch batch lists, straight from the sampler (what each rank iterates).
    s_a = GoldLengthGroupedBatchSampler(gold, BATCH, drop_last=True, shuffle=True, seed=SEED)
    s_b = GoldLengthGroupedBatchSampler(gold, BATCH, drop_last=True, shuffle=True, seed=SEED)
    full_a, full_b = [], []
    for e in range(EPOCHS):
        random.seed(1000 + e)
        full_a.append(list(s_a))
        random.seed(2000 + e)
        full_b.append(list(s_b))

    r0 = rank_epochs(gold, 0, perturb_seed=11, seed=SEED)
    r1 = rank_epochs(gold, 1, perturb_seed=22, seed=SEED)
    r0_again = rank_epochs(gold, 0, perturb_seed=99, seed=SEED)

    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

    print(f"examples={n} batch={BATCH} ranks={NPROC} epochs={EPOCHS}")
    for e in range(EPOCHS):
        # accelerate keeps ranks in lockstep: with drop_last it drops a trailing odd batch.
        kept = full_a[e][: len(full_a[e]) - len(full_a[e]) % NPROC]
        in_full = [i for b in kept for i in b]
        total, uniq, missing = coverage(r0[e], r1[e], n)
        print(f"epoch {e+1}: full batches={len(full_a[e])} rank0={len(r0[e])} rank1={len(r1[e])} "
              f"indices seen={total} unique={uniq} never seen={missing} (drop_last leftovers={n - len(in_full)})")
        check("both ranks' samplers build the identical batch list", full_a[e] == full_b[e])
        check("rank batches are disjoint (no example twice)", total == uniq)
        check("ranks together cover every kept full batch exactly once", sorted(in_full) == sorted(i for b in r0[e] + r1[e] for i in b))
        check("both ranks get the same number of batches", len(r0[e]) == len(r1[e]))
        check("every batch has a single gold count", all(len({gold[i] for i in b}) == 1 for b in r0[e] + r1[e]))
        check("rerun with the same seed reproduces rank 0's order", r0[e] == r0_again[e])
        if e > 0:
            check("order differs from the previous epoch", full_a[e] != full_a[e - 1])

    # Negative control: old sampler, ranks' global RNGs diverge.
    o0 = rank_epochs(gold, 0, perturb_seed=11, sampler_cls=OldSampler)
    o1 = rank_epochs(gold, 1, perturb_seed=22, sampler_cls=OldSampler)
    total, uniq, missing = coverage(o0[0], o1[0], n)
    print(f"negative control (old sampler, epoch 1): seen={total} unique={uniq} "
          f"duplicated={total - uniq} ({(total - uniq) / n:.1%}) never seen={missing} ({missing / n:.1%})")
    check("control detects duplicated examples under the old behaviour", total > uniq)

    print("RESULT:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
