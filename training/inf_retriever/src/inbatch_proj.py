"""Frozen-doc-encoder multi-query model with a learned input and output layer (objective audit, 2026-10-15).

Kept in its own module so inbatch.py (and every run that uses it) is untouched. Selected only by
--io_proj, which finetuning_multi_proj.py adds; src/inference_utils.load_retriever picks this class
for checkpoints whose saved opt has io_proj set.

It mirrors the legacy src/model.py EmbeddingModel, which wrapped its LM in two bias-free linear layers:
  - output layer: the scored query is out_proj(h), h = the query encoder's last-token state;
  - input layer:  a fed-in vector v (teacher-forced gold or the model's own projected output) enters
    the LM as in_proj(normalize(v)), as legacy fed input_projection(gold) and
    input_projection(output_projection(h)).
Both live in the retriever's embedding space, so teacher golds (frozen doc embeddings) and the model's own
outputs go through the same input layer.

Initialization keeps step 0 equal to the frozen baseline except for input scale:
  - out_proj = identity, so out_proj(h) = h;
  - in_proj = s * identity, with s the mean norm of the token-embedding table, so a fed vector enters at token
    scale (the parameter-free --fed_norm token does the same per question) instead of norm ~195.

The output layer is applied by a forward hook on the query encoder (self.encoder): every query-side call
(forward and generate, training and retrieval) then pools a projected state. Documents are encoded by the
separate frozen self.doc_encoder and are not projected. A hook rather than a wrapper module keeps the
encoder's state-dict keys unchanged. The input layer replaces _fed_input, the single place where both
training and generate() build the next fed token.

--io_proj_mode (default both; saved in opt, so load_retriever rebuilds the same layers) keeps one layer only:
  - in:  input layer only; queries are scored on the encoder's own state h (as the base class);
  - out: output layer only; fed vectors enter raw, as the base class with --fed_norm none (norm ~195).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import inbatch


class EmbeddingModelFrozenDocEncProj(inbatch.EmbeddingModelFrozenDocEnc):

    def __init__(self, opt, retriever=None, tokenizer=None):
        super().__init__(opt, retriever, tokenizer)
        assert getattr(opt, "fed_norm", "none") == "none", "--io_proj replaces --fed_norm; set only one"
        self.io_proj_mode = getattr(opt, "io_proj_mode", "both")
        assert self.io_proj_mode in ("both", "in", "out"), self.io_proj_mode
        d = self.embedding.weight.size(1)
        with torch.no_grad():
            s = self.embedding.weight.float().norm(dim=-1).mean().item()
        self.in_proj_init_scale = s
        if self.io_proj_mode in ("both", "in"):
            self.in_proj = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                self.in_proj.weight.copy_(torch.eye(d) * s)
        if self.io_proj_mode in ("both", "out"):
            self.out_proj = nn.Linear(d, d, bias=False)
            with torch.no_grad():
                self.out_proj.weight.copy_(torch.eye(d))
            self.encoder.register_forward_hook(self._project_query_output)

    def _project_query_output(self, module, args, output):
        h = output.last_hidden_state
        output.last_hidden_state = self.out_proj(h.to(self.out_proj.weight.dtype)).to(h.dtype)
        return output

    def _fed_input(self, vec, q_embeds, q_mask):
        if self.io_proj_mode == "out":
            return super()._fed_input(vec, q_embeds, q_mask)
        x = F.normalize(vec.float(), dim=-1).to(self.in_proj.weight.dtype)
        return self.in_proj(x).to(q_embeds.dtype)
