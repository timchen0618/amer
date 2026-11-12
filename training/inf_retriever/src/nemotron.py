"""nvidia/Nemotron-3.5-Embed-1B-BF16 as the retriever of the frozen-doc-encoder setting (session N, 2026-11-12).

Everything Nemotron-specific lives here, so the inf-retriever code paths stay unchanged unless --retriever nemotron
is passed (training: finetuning_multi_nemo.py; retrieval: src/inference_utils.load_retriever reads opt.retriever).

Model facts (model card / config.json, checked 2026-11-12):
  * Mistral3Model (vision tower + Ministral-3 text backbone); text-only use needs only .language_model
    (Mistral3Model's text path is embed_tokens -> language_model, so the outputs are identical).
  * hidden size = embedding size = 2048; bidirectional attention (config is_causal false);
    mean pooling over all tokens (prompt included), then L2 normalization.
  * inputs: the chat template with the prefix as system message, which renders to
    "<prefix> " + text with '[' and '<' escaped by a following zero-width non-joiner; no BOS / EOS
    (vLLM example: add_special_tokens=False). Prefixes: "query:" and "passage:".
  * needs transformers >= 5.12 (model_type ministral3): the nemotron.ext3 overlay, not div.ext3 / nli.ext3.

Multi-query generation (EmbeddingModelDocEncNoProj.forward / generate) pools each step with self.last_token_pool;
NemotronMixin replaces it by the masked mean, so step j's embedding is the mean of the hidden states of the
question tokens and the j embeddings fed in so far (all positions attend to each other).
"""
import logging

import torch
import torch.nn.functional as F
import transformers

logger = logging.getLogger(__name__)

NEMOTRON_ID = "nvidia/Nemotron-3.5-Embed-1B-BF16"
INF_ID = "infly/inf-retriever-v1-1.5b"
EMB_DIM = {"inf": 1536, "nemotron": 2048}
QUERY_PREFIX = "query:"
PASSAGE_PREFIX = "passage:"
_ZWNJ = "‌"


def _escape(text):
    """The chat template's mark_user_text: '[' -> '[‌', '<' -> '<‌' (control-token escape)."""
    return text.replace("[", "[" + _ZWNJ).replace("<", "<" + _ZWNJ)


def format_input(prefix, text):
    """String equal to chat_template.jinja for [system: prefix, user: text] (checked against
    tokenizer.apply_chat_template in tools/nemo_smoke.py)."""
    return prefix.rstrip(" ") + " " + _escape(text)


def format_query(question):
    return format_input(QUERY_PREFIX, question)


def format_passage(passage_text):
    return format_input(PASSAGE_PREFIX, passage_text)


class NoSpecialTokens:
    """Tokenizer proxy: every encode call runs with add_special_tokens=False (the template has no BOS / EOS).
    batch_encode_plus is provided for the shared collators (transformers 5 may not have it)."""

    def __init__(self, tok):
        self.__dict__["_tok"] = tok

    def __call__(self, *a, **kw):
        kw["add_special_tokens"] = False
        return self._tok(*a, **kw)

    def batch_encode_plus(self, texts, **kw):
        return self(list(texts), **kw)

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __setattr__(self, name, value):
        setattr(self._tok, name, value)


def load_tokenizer():
    tok = transformers.AutoTokenizer.from_pretrained(NEMOTRON_ID)
    assert tok.pad_token is not None
    tok.padding_side = "right"
    return NoSpecialTokens(tok)


def load_backbone(grad_ckpt=True):
    """The text backbone (Ministral3Model) in bf16, bidirectional as configured."""
    full = transformers.AutoModel.from_pretrained(NEMOTRON_ID, dtype=torch.bfloat16)
    lm = full.language_model
    assert getattr(lm.config, "is_causal", None) is False, lm.config
    lm.config.use_cache = False
    del full
    if grad_ckpt:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return lm


def mean_pool(last_hidden_states, attention_mask):
    m = attention_mask.unsqueeze(-1).to(last_hidden_states.dtype)
    return (last_hidden_states * m).sum(1) / m.sum(1).clamp(min=1)


class NemotronMixin:
    """Put first in the bases of an inbatch model class: Nemotron backbone + tokenizer, mean pooling."""

    def __init__(self, opt, retriever=None, tokenizer=None):
        opt.retriever_model_id = NEMOTRON_ID  # saved in the checkpoint's opt
        super().__init__(opt, retriever, tokenizer)

    def _load_retriever(self, model_id):
        return load_backbone(), load_tokenizer()

    def last_token_pool(self, last_hidden_states, attention_mask):  # name kept: the shared code calls it
        return mean_pool(last_hidden_states, attention_mask)


def nemotron_class(base):
    cls = type(base.__name__, (NemotronMixin, base), {"__module__": __name__})
    cls.__qualname__ = "Nemotron" + base.__name__
    return cls


def use_nemotron_text_format():
    """Training data: queries and passages in Nemotron's input format (finetuning_data builds every query with
    get_detailed_instruct and every passage with format_passage)."""
    from src import finetuning_data
    fmt_p = finetuning_data.format_passage

    finetuning_data.get_detailed_instruct = lambda task, query: format_query(query)
    finetuning_data.format_passage = lambda p: format_passage(fmt_p(p))
    logger.warning("[nemotron] queries 'query: <q>', passages 'passage: <title> <text>' (no BOS / EOS)")
