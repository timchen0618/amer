"""Checks for the objective-audit options (CPU, single process, mock encoder; 2026-10-13).

See reports/objective_audit_2026-10-13/PLAN.md (D1, E1, E2, E5).

1. hungarian_seq with nothing fed in equals HungarianContrastiveLoss (loss and assignment).
2. Its matching never gives output t a gold fed in before t, and is the best such matching
   (brute force over permutations); the copy solution that plain Hungarian takes is refused.
3. hungarian_seq_used keeps exactly the used golds (fed in before t, or matched to an earlier output)
   as same-example negatives: equals a brute-force per-row cross-entropy.
4. hungarian_plus_single passes fed_gold to a seq multi term; build_loss / option wiring.
5. copy_stats: a copying model scores tf_copy_match = tf_copy_cos = 1; a repeating one
   free_repeat_cos = free_repeat_top1 = 1.
6. Model forward records fed_gold from the sampling draw; --detach_fed_output removes the gradient
   that reaches output 0 through the fed-back input; --fed_norm token feeds vectors at the question's
   mean token-embedding norm in training and generate(); the default path is unchanged.
7. Plain Hungarian now logs training accuracy.

Run from the repo root:  python training/inf_retriever/tests/test_objective_audit.py
"""
import itertools
import os
import sys
import types

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from src import inbatch  # noqa: E402

D = 8


def close(a, b, tol=1e-5):
    assert abs(float(a) - float(b)) < tol, (a, b)


def test_seq_equals_hungarian_when_nothing_fed():
    torch.manual_seed(0)
    b, k, d = 4, 5, 16
    q, p, n = torch.randn(b, k, d), torch.randn(b, k, d), torch.randn(b, k, d)
    ref = inbatch.HungarianContrastiveLoss()
    seq = inbatch.HungarianSeqLoss(siblings="negative")
    l_ref, _ = ref(q, p, n, stats_prefix="t")
    l_seq, _ = seq(q, p, n, stats_prefix="t")
    close(l_ref, l_seq, 1e-5)
    assert torch.equal(ref.last_assignment, seq.last_assignment)
    l_seq0, _ = seq(q, p, n, stats_prefix="t", fed_gold=torch.zeros(b, k, dtype=torch.bool))
    close(l_ref, l_seq0, 1e-5)


def brute_best(cost, forbidden):
    k = cost.shape[0]
    best, arg = -np.inf, None
    for perm in itertools.permutations(range(k)):
        if any(forbidden[t, perm[t]] for t in range(k)):
            continue
        v = sum(cost[t, perm[t]] for t in range(k))
        if v > best:
            best, arg = v, perm
    return best, arg


def test_seq_matching_respects_fed_golds_and_is_optimal():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    b, k, d = 6, 5, 12
    q, p, n = torch.randn(b, k, d), torch.randn(b, k, d), torch.randn(b, k, d)
    fed = torch.from_numpy(rng.random((b, k)) < 0.6)
    fed[:, -1] = False
    loss = inbatch.HungarianSeqLoss(siblings="negative", normalize_embeddings=True)
    loss(q, p, n, fed_gold=fed)
    a = loss.last_assignment.numpy()
    qn, pn = F.normalize(q, dim=-1), F.normalize(p, dim=-1)
    for i in range(b):
        cost = (qn[i] @ pn[i].T).numpy()
        forbidden = np.tril(np.ones((k, k), dtype=bool), -1) & fed[i].numpy()[None, :]
        assert not any(forbidden[t, a[i, t]] for t in range(k)), (i, a[i], fed[i])
        best, _ = brute_best(cost, forbidden)
        close(sum(cost[t, a[i, t]] for t in range(k)), best, 1e-4)


def test_copy_solution_refused():
    torch.manual_seed(2)
    b, k, d = 3, 4, 16
    p = torch.randn(b, k, d)
    q = torch.randn(b, k, d)
    q[:, 1:] = p[:, :-1]                      # output t + 1 = gold t, the teacher-forced copy
    n = torch.randn(b, k, d)
    fed = torch.ones(b, k, dtype=torch.bool); fed[:, -1] = False
    plain = inbatch.HungarianContrastiveLoss()
    plain(q, p, n)
    assert (plain.last_assignment[:, 1:] == torch.arange(k - 1)).all()   # plain takes the copy
    seq = inbatch.HungarianSeqLoss(siblings="negative")
    seq(q, p, n, fed_gold=fed)
    # every gold fed before t is forbidden, so with everything fed output t must take gold t
    assert (seq.last_assignment == torch.arange(k)).all(), seq.last_assignment
    stats = inbatch.copy_stats(q, p, fed, plain.last_assignment)
    close(stats["tf_copy_match"][0] / stats["tf_copy_match"][1], 1.0)
    close(stats["tf_copy_cos"][0] / stats["tf_copy_cos"][1], 1.0, 1e-4)
    stats = inbatch.copy_stats(q, p, fed, seq.last_assignment)
    close(stats["tf_copy_match"][0], 0.0)


def test_seq_used_matches_brute_force():
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    b, k, d, tau = 3, 4, 10, 0.05
    q, p, n = torch.randn(b, k, d), torch.randn(b, k, d), torch.randn(b, k, d)
    fed = torch.from_numpy(rng.random((b, k)) < 0.5); fed[:, -1] = False
    loss_mod = inbatch.HungarianSeqLoss(temperature=tau, siblings="used")
    loss, stats = loss_mod(q, p, n, stats_prefix="t", fed_gold=fed)
    a = loss_mod.last_assignment
    qn, pn, nn_ = (F.normalize(x, dim=-1).reshape(b * k, d) for x in (q, p, n))
    pool = torch.cat([pn, nn_])
    tot = 0.0
    for i in range(b):
        for t in range(k):
            used = {int(a[i, s]) for s in range(t)} | {j for j in range(t) if fed[i, j]} | {int(a[i, t])}
            keep = [c for c in range(2 * b * k) if not (k * i <= c < k * (i + 1)) or (c - k * i) in used]
            logits = (qn[k * i + t] @ pool[keep].T) / tau
            target = keep.index(k * i + int(a[i, t]))
            tot += float(-F.log_softmax(logits, dim=0)[target])
    close(loss, tot / (b * k), 1e-4)
    loss.backward() if loss.requires_grad else None
    for name in ("accuracy", "accuracy_strict", "fed_gold_frac"):
        assert f"t/{name}" in stats


def test_seq_gradient_finite():
    torch.manual_seed(4)
    q = torch.randn(2, 3, 8, requires_grad=True)
    p, n = torch.randn(2, 3, 8), torch.randn(2, 3, 8)
    fed = torch.tensor([[True, True, False], [False, True, False]])
    for sib in ("negative", "used"):
        q.grad = None
        loss, _ = inbatch.HungarianSeqLoss(siblings=sib)(q, p, n, fed_gold=fed)
        loss.backward()
        assert torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0


def opt_ns(**kw):
    o = dict(norm_doc=True, norm_query=True, label_smoothing=0.0, loss_fn="hungarian", training_mode="multi",
             temperature=0.05, force_causal=False, retriever_model_id="mock", use_lora=False)
    o.update(kw)
    return types.SimpleNamespace(**o)


def test_hps_with_seq_multi_and_build_loss():
    torch.manual_seed(5)
    b, k, d = 3, 4, 8
    q, p, n = torch.randn(b, k, d), torch.randn(b, k, d), torch.randn(b, k, d)
    fed = torch.ones(b, k, dtype=torch.bool); fed[:, -1] = False
    hps = inbatch.build_loss(opt_ns(loss_fn="hungarian_plus_single", hps_multi_loss="hungarian_seq_used"))
    assert hps.uses_feed_info and isinstance(hps.multi, inbatch.HungarianSeqLoss) and hps.multi.siblings == "used"
    torch.manual_seed(9); l1, st = hps(q, p, n, stats_prefix="t", fed_gold=fed)
    seq = inbatch.HungarianSeqLoss(siblings="used")
    lm = seq._core_forward(q, p, n, fed_gold=fed)
    close(st["t/loss_multi"][0], lm, 1e-5)
    assert torch.equal(hps.last_assignment, seq.last_assignment)
    assert "t/multi_accuracy" in st
    plain_hps = inbatch.build_loss(opt_ns(loss_fn="hungarian_plus_single"))
    assert not plain_hps.uses_feed_info and isinstance(plain_hps.multi, inbatch.HungarianContrastiveLoss)
    for name, sib in (("hungarian_seq", "negative"), ("hungarian_seq_used", "used")):
        m = inbatch.build_loss(opt_ns(loss_fn=name))
        assert isinstance(m, inbatch.HungarianSeqLoss) and m.siblings == sib


def test_options_parse():
    from src.options import Options
    argv = sys.argv
    try:
        sys.argv = ["x", "--training_mode", "multi", "--loss_fn", "hungarian_plus_single",
                    "--hps_multi_loss", "hungarian_seq", "--detach_fed_output", "--fed_norm", "token",
                    "--select_metric", "cover5"]
        o = Options().parse()
        assert o.hps_multi_loss == "hungarian_seq" and o.detach_fed_output and o.fed_norm == "token"
        assert o.select_metric == "cover5"
        sys.argv = ["x"]
        o = Options().parse()
        assert (o.hps_multi_loss, o.detach_fed_output, o.fed_norm, o.select_metric) == ("hungarian", False, "none", "mrr")
    finally:
        sys.argv = argv


def test_copy_stats_repeat():
    torch.manual_seed(6)
    b, k, d = 2, 4, 8
    p = torch.randn(b, k, d)
    q = torch.randn(b, 1, d).repeat(1, k, 1)              # one query repeated
    fed = torch.zeros(b, k, dtype=torch.bool)             # free-running everywhere
    st = inbatch.copy_stats(q, p, fed)
    assert "tf_copy_cos" not in st
    close(st["free_repeat_cos"][0] / st["free_repeat_cos"][1], 1.0, 1e-5)
    close(st["free_repeat_top1"][0] / st["free_repeat_top1"][1], 1.0)


def test_plain_hungarian_logs_accuracy():
    torch.manual_seed(7)
    b, k, d = 3, 4, 16
    p = torch.randn(b, k, d)
    q = p[:, torch.randperm(k)] + 0.01 * torch.randn(b, k, d)
    _, st = inbatch.HungarianContrastiveLoss()(q, p, torch.randn(b, k, d), stats_prefix="t")
    close(st["t/accuracy_strict"][0], 100.0)
    close(st["t/accuracy"][0], 100.0)


# ---- model forward / generate with a mock encoder ----

class Out:
    def __init__(self, h):
        self.last_hidden_state = h


class MockEncoder(nn.Module):
    """Position-wise (no attention mixing): output at a position depends only on that position's input."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(50, D)
        self.lin = nn.Linear(D, D)
        self.inputs = []

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None, **kw):
        x = self.emb(input_ids) if input_ids is not None else inputs_embeds
        if input_ids is None:
            self.inputs.append(x.detach().clone())
        return Out(self.lin(x))


def build(**kw):
    return inbatch.EmbeddingModelDocEncNoProj(opt_ns(**kw), MockEncoder(), object())


def batch(b=3, k=4, L=5):
    q = torch.randint(1, 50, (b, L)); m = torch.ones(b, L, dtype=torch.long)
    pos = torch.arange(L).repeat(b, 1)
    kt = torch.randint(1, 50, (2 * b * k, L)); km = torch.ones(2 * b * k, L, dtype=torch.long)
    return dict(q_tokens=q, q_mask=m, q_position_ids=pos, k_tokens=kt, k_mask=km)


class Capture(nn.Module):
    """Stand-in loss: records its inputs and returns a chosen function of the outputs."""

    def __init__(self, fn, uses_feed_info=False):
        super().__init__()
        self.fn, self.uses_feed_info, self.seen = fn, uses_feed_info, {}

    def forward(self, outputs, pos, neg, stats_prefix="", iter_stats=None, **kw):
        self.seen = dict(kw, outputs=outputs)
        return self.fn(outputs), iter_stats


def test_forward_records_fed_gold():
    torch.manual_seed(8)
    m = build()
    m.loss_fct = Capture(lambda o: o.sum(), uses_feed_info=True)
    m(**batch(), sampling_rate=0.0)
    fed = m.loss_fct.seen["fed_gold"]
    assert fed[:, :-1].all() and not fed[:, -1].any()
    m(**batch(), sampling_rate=1.0)
    assert not m.loss_fct.seen["fed_gold"].any()
    m.loss_fct = Capture(lambda o: o.sum(), uses_feed_info=False)
    m(**batch(), sampling_rate=0.5)
    assert "fed_gold" not in m.loss_fct.seen


def grad_through_question(detach):
    torch.manual_seed(10)
    m = build(detach_fed_output=detach)
    m.loss_fct = Capture(lambda o: o[:, 1:].sum())          # only outputs 1..k-1 are scored
    m.zero_grad()
    loss, _ = m(**batch(), sampling_rate=1.0)
    loss.backward()
    return m.encoder.emb.weight.grad                         # question tokens reach outputs 1.. only via output 0


def test_detach_fed_output():
    g_on = grad_through_question(False)
    g_off = grad_through_question(True)
    assert g_on is not None and g_on.abs().sum() > 0
    assert g_off is None or g_off.abs().sum() == 0


def test_fed_norm_token():
    torch.manual_seed(11)
    bt = batch()
    m = build(fed_norm="token")
    m.loss_fct = Capture(lambda o: o.sum())
    m(**bt, sampling_rate=0.5)
    tok = m.encoder.emb(bt["q_tokens"]).norm(dim=-1).mean(dim=1)              # (b,) mean token norm
    for x in m.encoder.inputs[1:]:
        assert torch.allclose(x[:, -1].norm(dim=-1), tok, atol=1e-4)
    m.encoder.inputs.clear()
    m.generate(bt["q_tokens"], bt["q_mask"], bt["q_position_ids"], max_new_tokens=4)
    for x in m.encoder.inputs[1:]:
        assert torch.allclose(x[:, -1].norm(dim=-1), tok, atol=1e-4)


def test_default_path_unchanged():
    """Flags off: the fed vector is the raw output / gold, and the rename is the only stat change."""
    torch.manual_seed(12)
    bt = batch()
    m = build()
    m.loss_fct = Capture(lambda o: o.sum())
    _, st = m(**bt, sampling_rate=1.0, stats_prefix="t")
    outs = m.loss_fct.seen["outputs"]
    for t, x in enumerate(m.encoder.inputs[1:]):
        assert torch.allclose(x[:, -1], outs[:, t].detach(), atol=1e-6)
    assert "t/step_0_next_fed_gold_cos" in st and not any("teacher_cos_sim" in key for key in st)
    assert "t/free_repeat_cos" in st and "t/tf_copy_cos" not in st


if __name__ == "__main__":
    tests = [v for k_, v in list(globals().items()) if k_.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"all {len(tests)} passed")
