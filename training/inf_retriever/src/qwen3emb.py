"""Qwen/Qwen3-Embedding-0.6B as the retriever of the frozen-doc-encoder setting (session N, 2026-11-15).

The Nemotron study (src/nemotron.py) repeated with Qwen3-Embedding-0.6B. Unlike Nemotron, this model is close to
inf-retriever, so the inf code paths are used as they are; only the model id and the query format change.

Model facts (config.json, tokenizer.json, 1_Pooling, config_sentence_transformers.json; checked 2026-11-15):
  * Qwen3Model (model_type qwen3, transformers >= 4.51: div.ext3 4.56 and nli.ext3 4.57 both have it), hidden size =
    embedding size = 1024, 28 layers, bf16.
  * causal attention (Qwen3Model always builds a causal mask) + last-token pooling, then L2 normalization: the inf
    path's last_token_pool as is. No --force_causal needed (and none allowed: the per-call is_causal kwarg is
    inf-retriever's remote code).
  * the tokenizer appends <|endoftext|> (id 151643, also the pad token) to every sequence, as inf-retriever's does;
    the pooled token is that appended token. (Its eos_token is <|im_end|>, but nothing here inserts eos by hand.)
  * queries: 'Instruct: {task}\\nQuery:{query}' (no space after 'Query:', unlike inf-retriever's 'Query: ');
    documents: the raw text, no prompt.
"""
import logging

import torch
import torch.nn.functional as F
import transformers

logger = logging.getLogger(__name__)

QWEN3_ID = "Qwen/Qwen3-Embedding-0.6B"
EMB_DIM = 1024
# finetuning_data's training task string (the same one the inf-retriever runs use)
TASK = "Given a query, retrieve relevant passages that answer the query"


def get_detailed_instruct(task_description, query):
    return f"Instruct: {task_description}\nQuery:{query}"


def format_query(question, task=TASK):
    return get_detailed_instruct(task, question)


def load_tokenizer():
    tok = transformers.AutoTokenizer.from_pretrained(QWEN3_ID)
    assert tok.pad_token == "<|endoftext|>", tok.pad_token
    tok.padding_side = "right"
    return tok


def load_backbone():
    lm = transformers.AutoModel.from_pretrained(QWEN3_ID, torch_dtype=torch.bfloat16, use_cache=False)
    assert lm.config.model_type == "qwen3", lm.config.model_type
    return lm


def last_token_pool(last_hidden_states, attention_mask):
    """inbatch's last_token_pool (left or right padding)."""
    if (attention_mask[:, -1].sum() == attention_mask.shape[0]):
        return last_hidden_states[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_states[torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device), lengths]


@torch.no_grad()
def embed_texts(texts, model, tok, max_length, batch_size=64):
    """L2-normalized last-token embeddings (float32 numpy), right padding, bf16 forward."""
    import numpy as np
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        b = tok(texts[i:i + batch_size], max_length=max_length, truncation=True, padding=True,
                return_tensors="pt").to("cuda")
        h = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        out[i:i + len(h)] = F.normalize(last_token_pool(h, b["attention_mask"]).float(), dim=-1).cpu().numpy()
    return out


class Qwen3Mixin:
    """Put first in the bases of an inbatch model class: the Qwen3-Embedding model id; everything else is inherited
    (inbatch._load_retriever: AutoModel bf16 + gradient checkpointing; last-token pooling; causal by the model)."""

    def __init__(self, opt, retriever=None, tokenizer=None):
        assert not getattr(opt, "force_causal", False), "Qwen3-Embedding is already causal; drop --force_causal"
        opt.retriever_model_id = QWEN3_ID  # finetuning_multi.main sets inf-retriever's id; saved in the checkpoint
        super().__init__(opt, retriever, tokenizer)
        enc = self.encoder.get_base_model() if hasattr(self.encoder, "get_base_model") else self.encoder
        assert enc.config.model_type == "qwen3", enc.config.model_type


def qwen3_class(base):
    cls = type(base.__name__, (Qwen3Mixin, base), {"__module__": __name__})
    cls.__qualname__ = "Qwen3" + base.__name__
    return cls


def use_qwen3_text_format():
    """Training data: queries in Qwen3-Embedding's format; passages unchanged ('<title> <text>', no prompt)."""
    from src import finetuning_data
    finetuning_data.get_detailed_instruct = get_detailed_instruct
    logger.warning("[qwen3] queries 'Instruct: <task>\\nQuery:<q>', passages '<title> <text>' (+ <|endoftext|>)")
