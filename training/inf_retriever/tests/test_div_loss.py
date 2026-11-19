"""Checks for --loss_fn hungarian_masked_div and --sampling_saturate_frac (CPU, single process).

1. The diversity penalty equals a brute-force loop over examples and pairs, for 3-D and flattened positives.
2. Its assignment is the masked loss's own: outputs that are a permuted copy of the golds give penalty 0
   (each query pair is exactly as alike as its golds), and the contrastive part equals hungarian_masked.
3. Identical query embeddings are penalised and the gradient pushes them apart; pairs less alike than
   their golds are not penalised; k = 1 gives 0.
4. build_loss wiring, and the sampling-rate schedule for several saturation fractions.

Run from the repo root:  python training/inf_retriever/tests/test_div_loss.py
"""
import os
import sys
import types

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from src import inbatch  # noqa: E402


def close(a, b, tol=1e-5):
    assert abs(float(a) - float(b)) < tol, (a, b)


def brute_penalty(q, g, assign):
    b, k, _ = q.shape
    tot, n = 0.0, 0
    for i in range(b):
        for x in range(k):
            for y in range(k):
                if x == y:
                    continue
                cq = F.cosine_similarity(q[i, x], q[i, y], dim=0)
                cg = F.cosine_similarity(g[i, assign[i, x]], g[i, assign[i, y]], dim=0)
                tot += max(0.0, float(cq - cg)); n += 1
    return tot / n


def test_penalty_matches_brute_force():
    torch.manual_seed(0)
    b, k, d = 4, 5, 16
    q, p, n = torch.randn(b, k, d), torch.randn(b, k, d), torch.randn(b, k, d)
    loss = inbatch.HungarianMaskedDivLoss(div_weight=1.0)
    total, stats = loss(q, p, n, stats_prefix="t")
    assign = loss.last_assignment
    ref = brute_penalty(q, p, assign)
    close(stats["t/div_penalty"][0], ref)
    close(total, stats["t/loss_masked"][0] + ref, 1e-4)
    # flattened positives (as the real model passes them) give the same numbers
    total2, stats2 = inbatch.HungarianMaskedDivLoss(div_weight=1.0)(q, p.reshape(b * k, d), n.reshape(b * k, d), stats_prefix="t")
    close(total, total2, 1e-5)
    close(stats["t/div_penalty"][0], stats2["t/div_penalty"][0])


def test_masked_part_unchanged_and_assignment_shared():
    torch.manual_seed(1)
    b, k, d = 3, 4, 8
    q, p, n = torch.randn(b, k, d), torch.randn(b, k, d), torch.randn(b, k, d)
    base, _ = inbatch.HungarianMaskedContrastiveLoss()(q, p, n)
    div = inbatch.HungarianMaskedDivLoss(div_weight=0.0)
    total, stats = div(q, p, n, stats_prefix="t")
    close(base, total)
    close(base, stats["t/loss_masked"][0])
    # outputs = a permuted copy of the golds: Hungarian recovers the permutation, penalty is exactly 0
    perm = torch.stack([torch.randperm(k) for _ in range(b)])
    q2 = torch.gather(p, 1, perm.unsqueeze(-1).expand(b, k, d))
    div2 = inbatch.HungarianMaskedDivLoss(div_weight=1.0)
    _, stats2 = div2(q2, p, n, stats_prefix="t")
    assert torch.equal(div2.last_assignment, perm), (div2.last_assignment, perm)
    close(stats2["t/div_penalty"][0], 0.0)


def test_direction_and_edge_cases():
    torch.manual_seed(2)
    b, k, d = 2, 3, 8
    p, n = torch.randn(b, k, d), torch.randn(b, k, d)
    # identical query embeddings: penalised, and the gradient reduces their pairwise cosine
    base = torch.randn(b, 1, d)
    q = (base.repeat(1, k, 1) + 1e-3 * torch.randn(b, k, d)).requires_grad_(True)
    loss = inbatch.HungarianMaskedDivLoss(div_weight=1.0)
    pen = loss.diversity_penalty(q, p, torch.arange(k).repeat(b, 1))
    assert pen > 0.1, pen
    pen.backward()
    with torch.no_grad():
        q_new = q - 0.5 * q.grad
        before = F.cosine_similarity(q[:, 0], q[:, 1], dim=-1).mean()
        after = F.cosine_similarity(q_new[:, 0], q_new[:, 1], dim=-1).mean()
    assert after < before, (before, after)
    # orthogonal queries, golds all alike: no penalty
    q_orth = torch.eye(d)[:k].repeat(b, 1, 1)
    g_alike = torch.ones(b, k, d) + 0.01 * torch.randn(b, k, d)
    close(loss.diversity_penalty(q_orth, g_alike, torch.arange(k).repeat(b, 1)), 0.0)
    # k = 1
    close(loss.diversity_penalty(torch.randn(b, 1, d), torch.randn(b, 1, d), torch.zeros(b, 1, dtype=torch.long)), 0.0)


def test_build_loss_and_schedule():
    opt = types.SimpleNamespace(loss_fn="hungarian_masked_div", training_mode="multi", temperature=0.05, div_weight=2.0)
    l = inbatch.build_loss(opt)
    assert isinstance(l, inbatch.HungarianMaskedDivLoss) and l.div_weight == 2.0
    try:
        inbatch.build_loss(types.SimpleNamespace(loss_fn="hungarian_masked_div", training_mode="standard_org_q", temperature=0.05))
        raise AssertionError("single mode should be rejected")
    except ValueError:
        pass
    # the expression used in finetuning_multi.py
    rate = lambda step, frac, total=2500, full=False: 1.0 if full else min(1.0, step / (frac * total))
    close(rate(1250, 1.0), 0.5); close(rate(2500, 1.0), 1.0)
    close(rate(312.5, 0.25), 0.5); close(rate(625, 0.25), 1.0); close(rate(2000, 0.25), 1.0)
    close(rate(625, 0.5), 0.5); close(rate(1250, 0.5), 1.0)
    close(rate(0, 0.5, full=True), 1.0)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("all passed")
