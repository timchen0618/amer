"""Checks for the joint-doc-encoder input/output layers (src/inbatch_proj.EmbeddingModelDocEncProj), CPU, mock encoder.

1. The output layer projects query calls (inputs_embeds) only: documents, in training and through a direct
   encoder(input_ids=...) call as gen_embed_new.py makes, stay in the encoder's own space.
2. At initialization the first query equals the base class's; fed vectors enter as in_proj(normalize(v)).
3. --io_proj_mode in / out build only that layer; the state dict round-trips through load_model_state.
4. The real entry path: finetuning_multi_proj's option parsing makes finetuning_multi build the joint proj
   class, and leaves src.inbatch's own globals alone (job 20622528 died on exactly this, 2026-10-22).

Run from the repo root:  python training/inf_retriever/tests/test_io_proj_joint.py
"""
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from src import inbatch, inbatch_proj  # noqa: E402
from test_wave3_frozen_causal import MockEncoder, make_opt, batch  # noqa: E402


def build(cls, seed=0, **kw):
    torch.manual_seed(seed)
    enc = MockEncoder("query")
    return cls(make_opt(fed_norm="none", **kw), enc, object())


def test_docs_unprojected_queries_projected():
    base = build(inbatch.EmbeddingModelDocEncNoProj)
    proj = build(inbatch_proj.EmbeddingModelDocEncProj)
    with torch.no_grad():
        proj.out_proj.weight.mul_(2.0)  # non-identity, so a projection would show
    q, m, pos, kt, km = batch()
    pb, nb = base.encode_documents(kt, km)
    pp, np_ = proj.encode_documents(kt, km)
    assert torch.allclose(pb, pp) and torch.allclose(nb, np_), "documents must not be projected"
    direct_b = base.encoder(input_ids=kt, attention_mask=km).last_hidden_state
    direct_p = proj.encoder(input_ids=kt, attention_mask=km).last_hidden_state
    assert torch.allclose(direct_b, direct_p), "corpus embedding (gen_embed_new.py path) must not be projected"
    gb = base.generate(q, m, pos, max_new_tokens=1)
    gp = proj.generate(q, m, pos, max_new_tokens=1)
    assert torch.allclose(gp, 2 * gb, atol=1e-5), "query outputs must be projected"


def test_init_and_fed_input():
    base = build(inbatch.EmbeddingModelDocEncNoProj)
    proj = build(inbatch_proj.EmbeddingModelDocEncProj)
    q, m, pos, kt, km = batch()
    assert torch.allclose(base.generate(q, m, pos, 1), proj.generate(q, m, pos, 1)), "identity init: query 0 unchanged"
    v = torch.randn(2, 1, proj.embedding.weight.size(1))
    fed = proj._fed_input(v, proj.embedding(q), m)
    expect = F.normalize(v, dim=-1) * proj.in_proj_init_scale
    assert torch.allclose(fed, expect, atol=1e-5)
    loss, _ = proj(q, m, pos, kt, km, sampling_rate=0.5)
    loss.backward()
    assert proj.in_proj.weight.grad is not None and proj.out_proj.weight.grad is not None


def test_modes_and_state_dict():
    pin = build(inbatch_proj.EmbeddingModelDocEncProj, io_proj_mode="in")
    pout = build(inbatch_proj.EmbeddingModelDocEncProj, io_proj_mode="out")
    assert hasattr(pin, "in_proj") and not hasattr(pin, "out_proj")
    assert hasattr(pout, "out_proj") and not hasattr(pout, "in_proj")
    a = build(inbatch_proj.EmbeddingModelDocEncProj, seed=0)
    with torch.no_grad():
        a.out_proj.weight.add_(0.1); a.in_proj.weight.add_(0.1)
    b = build(inbatch_proj.EmbeddingModelDocEncProj, seed=1)
    inbatch.load_model_state(b, a.state_dict())
    q, m, pos, kt, km = batch()
    assert torch.allclose(a.generate(q, m, pos, 3), b.generate(q, m, pos, 3))


def test_entry_point_swap():
    sys.path.insert(0, os.path.join(HERE, ".."))
    import finetuning_multi
    import finetuning_multi_proj
    base_cls = inbatch.EmbeddingModelDocEncNoProj
    argv = sys.argv
    sys.argv = ["finetuning_multi_proj.py", "--training_mode", "multi", "--io_proj", "--loss_fn", "hungarian_seq"]
    try:
        finetuning_multi_proj.ProjOptions().parse()
    finally:
        sys.argv = argv
    assert inbatch.EmbeddingModelDocEncNoProj is base_cls, "src.inbatch globals must stay untouched"
    cls = finetuning_multi.inbatch.EmbeddingModelDocEncNoProj
    assert cls is inbatch_proj.EmbeddingModelDocEncProj
    m = build(cls)  # the constructor call that failed in job 20622528
    assert hasattr(m, "in_proj") and hasattr(m, "out_proj")
    assert finetuning_multi.inbatch.EmbeddingModelFrozenDocEnc is inbatch.EmbeddingModelFrozenDocEnc


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
