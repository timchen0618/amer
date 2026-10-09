"""Checks for wave 3 (frozen doc encoder x causal query attention), CPU, single process, mock encoder.

1. --force_causal adds is_causal=True to every QUERY-encoder call (training forward and generate(),
   multi- and single-query, joint and frozen classes) and to no document-encoder call; without it no
   query call gets an is_causal kwarg (the bidirectional runs' calls are unchanged).
2. The frozen multi-query class inherits forward()/generate() from the joint class (no copies left)
   and encodes documents with the frozen copy only.
3. utils.save_state_dict drops doc_encoder.* keys; inbatch.load_model_state accepts their absence
   and nothing else; utils.set_optim leaves the frozen copy out of the optimizer.

Run from the repo root:  python training/inf_retriever/tests/test_wave3_frozen_causal.py
"""
import os
import sys
import tempfile
import types

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from src import inbatch, utils  # noqa: E402

D = 8


class Out:
    def __init__(self, h):
        self.last_hidden_state = h


class MockEncoder(nn.Module):
    """Records the kwargs of every call; returns a deterministic function of the inputs."""

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.emb = nn.Embedding(50, D)
        self.lin = nn.Linear(D, D)
        self.calls = []

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, position_ids=None, **kw):
        self.calls.append({"kind": "doc" if input_ids is not None else "query", **kw})
        x = self.emb(input_ids) if input_ids is not None else inputs_embeds
        return Out(self.lin(x))


def make_opt(**kw):
    o = dict(norm_doc=True, norm_query=True, label_smoothing=0.0, loss_fn="hungarian", training_mode="multi",
             temperature=0.05, force_causal=False, retriever_model_id="mock", use_lora=False)
    o.update(kw)
    return types.SimpleNamespace(**o)


def build(cls, **kw):
    # the frozen classes load their doc-encoder copy through _load_retriever: return a second mock
    orig = cls._load_retriever
    cls._load_retriever = lambda self, mid: (MockEncoder("frozen_doc"), None)
    try:
        return cls(make_opt(**kw), MockEncoder("query"), object())
    finally:
        cls._load_retriever = orig


def batch(b=2, k=3, L=4):
    q = torch.randint(1, 50, (b, L)); m = torch.ones(b, L, dtype=torch.long)
    pos = torch.arange(L).repeat(b, 1)
    kt = torch.randint(1, 50, (2 * b * k, L)); km = torch.ones(2 * b * k, L, dtype=torch.long)
    return q, m, pos, kt, km


def query_doc_calls(model):
    calls = list(model.encoder.calls)
    if hasattr(model, "doc_encoder"):
        calls += model.doc_encoder.calls
    return [c for c in calls if c["kind"] == "query"], [c for c in calls if c["kind"] == "doc"]


def test_causal_kwarg_routing():
    for cls, mode in ((inbatch.EmbeddingModelDocEncNoProj, "multi"), (inbatch.EmbeddingModelFrozenDocEnc, "multi"),
                      (inbatch.EmbeddingModelDocEncNoProjSingleQuery, "standard_org_q"),
                      (inbatch.EmbeddingModelFrozenDocEncSingleQuery, "standard_org_q")):
        for causal in (False, True):
            m = build(cls, force_causal=causal, training_mode=mode,
                      loss_fn="hungarian" if mode == "multi" else "contrastive")
            q, qm, pos, kt, km = batch(k=3 if mode == "multi" else 1)
            m(q, qm, pos, kt, km, sampling_rate=0.5)
            if mode == "multi":
                m.generate(q, qm, pos, max_new_tokens=4)
            qc, dc = query_doc_calls(m)
            assert qc, cls
            for c in qc:
                assert (c.get("is_causal") is True) if causal else ("is_causal" not in c), (cls, causal, c)
            for c in dc:
                assert "is_causal" not in c, (cls, c)
            if hasattr(m, "doc_encoder"):  # frozen: documents only through the frozen copy
                assert all(c["kind"] == "query" for c in m.encoder.calls), cls
                assert m.doc_encoder.calls and all(c["kind"] == "doc" for c in m.doc_encoder.calls), cls


def test_frozen_inherits_loop():
    for name in ("forward", "generate"):
        assert getattr(inbatch.EmbeddingModelFrozenDocEnc, name) is getattr(inbatch.EmbeddingModelDocEncNoProj, name), name
    assert inbatch.EmbeddingModelFrozenDocEncSingleQuery.forward is inbatch.EmbeddingModelDocEncNoProjSingleQuery.forward


def test_save_load_optim():
    m = build(inbatch.EmbeddingModelFrozenDocEnc)
    sd = m.state_dict()
    assert any(k.startswith("doc_encoder.") for k in sd)
    sched = types.SimpleNamespace(state_dict=lambda: {})
    with tempfile.TemporaryDirectory() as tmp:
        utils.save_state_dict(sd, None, sched, 7, make_opt(), tmp, "step-7")
        saved = torch.load(os.path.join(tmp, "checkpoint", "step-7", "checkpoint.pth"), weights_only=False)["model"]
    assert saved and not any(k.startswith("doc_encoder.") for k in saved)
    assert set(saved) == {k for k in sd if not k.startswith("doc_encoder.")}
    m2 = build(inbatch.EmbeddingModelFrozenDocEnc)
    missing = inbatch.load_model_state(m2, saved)
    assert missing and all(k.startswith("doc_encoder.") for k in missing)
    for k, v in saved.items():
        assert torch.equal(m2.state_dict()[k], v), k
    # any other missing or unexpected key still fails
    for bad in ({k: v for k, v in list(saved.items())[1:]}, {**saved, "extra.weight": torch.zeros(1)}):
        try:
            inbatch.load_model_state(m2, bad); raise AssertionError("should fail")
        except RuntimeError:
            pass
    # joint model: no doc_encoder keys, nothing dropped, strict load
    j = build(inbatch.EmbeddingModelDocEncNoProj)
    with tempfile.TemporaryDirectory() as tmp:
        utils.save_state_dict(j.state_dict(), None, sched, 1, make_opt(), tmp, "step-1")
        sj = torch.load(os.path.join(tmp, "checkpoint", "step-1", "checkpoint.pth"), weights_only=False)["model"]
    assert set(sj) == set(j.state_dict())
    assert inbatch.load_model_state(build(inbatch.EmbeddingModelDocEncNoProj), sj) == []
    # optimizer: frozen copy excluded, everything else included
    o = types.SimpleNamespace(optim="adamw", lr=1e-5, beta1=0.9, beta2=0.98, eps=1e-6, weight_decay=0.01,
                              warmup_steps=1, total_steps=10, lr_min_ratio=0.0, scheduler="linear")
    opt_, _ = utils.set_optim(o, m)
    in_opt = {id(p) for g in opt_.param_groups for p in g["params"]}
    for n, p in m.named_parameters():
        assert (id(p) in in_opt) == (not n.startswith("doc_encoder.")), n
    opt_j, _ = utils.set_optim(o, j)
    assert sum(len(g["params"]) for g in opt_j.param_groups) == len(list(j.parameters()))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
    print("all passed")
