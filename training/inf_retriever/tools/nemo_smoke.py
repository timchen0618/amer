"""GPU checks of src/nemotron.py against the model card (session N, 2026-11-12). Run in the nemotron.ext3 overlay:
  python training/inf_retriever/tools/nemo_smoke.py
"""
import os
import sys

import torch
import torch.nn.functional as F
import transformers

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src import nemotron  # noqa: E402

print("transformers", transformers.__version__, "torch", torch.__version__)
tok = nemotron.load_tokenizer()
raw = tok._tok

# 1. input format == chat template, and token ids without BOS / EOS
for prefix, text in [("query:", "What is NVIDIA NeMo Framework?"), ("passage:", "a [b] <c> [INST] </s> x")]:
    tmpl = raw.apply_chat_template([{"role": "system", "content": prefix}, {"role": "user", "content": text}],
                                   tokenize=False)
    tmpl_list = raw.apply_chat_template([{"role": "system", "content": prefix},
                                         {"role": "user", "content": [{"type": "text", "text": text}]}], tokenize=False)
    mine = nemotron.format_input(prefix, text)
    print(f"[fmt] {mine!r} == template: {mine == tmpl} (list content: {mine == tmpl_list})")
    assert mine == tmpl == tmpl_list
    ids_t = raw.apply_chat_template([{"role": "system", "content": prefix}, {"role": "user", "content": text}],
                                    tokenize=True, return_dict=True)["input_ids"]
    ids_m = tok([mine])["input_ids"][0]
    print(f"[ids] equal: {list(ids_t) == list(ids_m)}; first ids {ids_m[:6]}, bos={raw.bos_token_id}")
    assert list(ids_t) == list(ids_m)
    assert tok.batch_encode_plus([mine])["input_ids"][0] == ids_m

# 2. model card scores (text only): full model vs text backbone
lm = nemotron.load_backbone(grad_ckpt=False).cuda().eval()
full = transformers.AutoModel.from_pretrained(nemotron.NEMOTRON_ID, dtype=torch.bfloat16).cuda().eval()


@torch.no_grad()
def emb(model, texts, pad_side="right"):
    raw.padding_side = pad_side
    b = tok(texts, padding=True, return_tensors="pt").to("cuda")
    raw.padding_side = "right"
    kw = {}
    if pad_side == "left":
        kw["position_ids"] = (b["attention_mask"].cumsum(1) - 1).clamp(min=0)
    h = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], **kw).last_hidden_state
    return F.normalize(nemotron.mean_pool(h, b["attention_mask"]).float(), dim=-1)


q = [nemotron.format_query("What is NVIDIA NeMo Framework?")]
docs = [nemotron.format_passage("NeMo Framework offers comprehensive functionalities for developing both LLMs and "
                                "Multimodal Models."),
        nemotron.format_passage("A biological foundation model designed to analyze and generate DNA, RNA, and "
                                "protein sequences.")]
for name, m in [("backbone", lm), ("full", full)]:
    s = (emb(m, q) @ emb(m, docs).T)[0].tolist()
    print(f"[card] {name}: nemo_text {s[0]:.6f} (card 0.457723), biology {s[1]:.6f} (card 0.112573)")
    assert abs(s[0] - 0.457723) < 0.02 and abs(s[1] - 0.112573) < 0.02
del full

# 3. bidirectional: the first token's state depends on later tokens
b1 = tok([nemotron.format_query("who wrote hamlet")], return_tensors="pt").to("cuda")
b2 = tok([nemotron.format_query("who wrote hamlet and macbeth")], return_tensors="pt").to("cuda")
with torch.no_grad():
    h1 = lm(**b1).last_hidden_state[0, 0].float()
    h2 = lm(**b2).last_hidden_state[0, 0].float()
print(f"[bidir] first-token state change when appending tokens: {(h1 - h2).norm() / h1.norm():.4f} (0 = causal)")
assert (h1 - h2).norm() / h1.norm() > 1e-3

# 4. padding: right-padded and left-padded (+ position ids) batches equal unpadded
texts = [nemotron.format_query("who wrote hamlet"), nemotron.format_query("list of operas by verdi first staged in milan")]
single = torch.cat([emb(lm, [t]) for t in texts])
for side in ("right", "left"):
    d = (emb(lm, texts, side) - single).norm(dim=-1).max().item()
    print(f"[pad] {side}-padded batch vs unpadded: max |diff| {d:.4f}")
    assert d < 0.03

# 5. inputs_embeds path (multi model) == input_ids path
b = tok(texts, padding=True, return_tensors="pt").to("cuda")
with torch.no_grad():
    a = lm(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    e = lm(inputs_embeds=lm.get_input_embeddings()(b["input_ids"]), attention_mask=b["attention_mask"]).last_hidden_state
print(f"[embeds] inputs_embeds vs input_ids max diff {(a - e).abs().max().item():.5f}")

# 6. fp16 vs bf16 (retrieval used fp16 for inf-retriever)
lm16 = nemotron.load_backbone(grad_ckpt=False).cuda().eval().half()
with torch.no_grad():
    bb = tok(docs + texts, padding=True, return_tensors="pt").to("cuda")
    h16 = lm16(**bb).last_hidden_state
    ok = torch.isfinite(h16).all().item()
    e16 = F.normalize(nemotron.mean_pool(h16, bb["attention_mask"]).float(), dim=-1)
    e32 = F.normalize(nemotron.mean_pool(lm(**bb).last_hidden_state, bb["attention_mask"]).float(), dim=-1)
print(f"[fp16] finite {ok}; cosine fp16 vs bf16 min {(e16 * e32).sum(-1).min().item():.4f}")
print(f"hidden_size {lm.config.hidden_size}, layers {lm.config.num_hidden_layers}, "
      f"params {sum(p.numel() for p in lm.parameters()) / 1e9:.2f}B, attn {lm.config._attn_implementation}")
print("SMOKE OK")
