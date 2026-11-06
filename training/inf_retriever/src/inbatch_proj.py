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
  - mlp_in (frozen class only, Stage F 2026-11-06): 2-layer MLP input layer, no output layer. A fed vector enters as
    in_proj(x + mlp(x)), x = normalize(v), mlp = Linear -> GELU -> Linear with the last Linear zero-initialized,
    so step 0 equals the "in" mode (identity init) and the MLP learns a residual on top of it.
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
        assert self.io_proj_mode in ("both", "in", "out", "mlp_in"), self.io_proj_mode
        d = self.embedding.weight.size(1)
        with torch.no_grad():
            s = self.embedding.weight.float().norm(dim=-1).mean().item()
        self.in_proj_init_scale = s
        # --io_proj_init random (2026-10-22, paper-gap plan): keep nn.Linear's default init, as the legacy
        # src/model.py projections did, instead of identity / s * identity.
        identity = getattr(opt, "io_proj_init", "identity") == "identity"
        if self.io_proj_mode in ("both", "in", "mlp_in"):
            self.in_proj = nn.Linear(d, d, bias=False)
            if identity:
                with torch.no_grad():
                    self.in_proj.weight.copy_(torch.eye(d) * s)
        if self.io_proj_mode == "mlp_in":
            self.in_mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
            with torch.no_grad():  # zero residual at step 0
                self.in_mlp[2].weight.zero_()
                self.in_mlp[2].bias.zero_()
        if self.io_proj_mode in ("both", "out"):
            self.out_proj = nn.Linear(d, d, bias=False)
            if identity:
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
        if self.io_proj_mode == "mlp_in":
            x = x + self.in_mlp(x)
        return self.in_proj(x).to(q_embeds.dtype)


class EmbeddingModelDocEncProj(inbatch.EmbeddingModelDocEncNoProj):
    """EmbeddingModelFrozenDocEncProj for the JOINT doc encoder (Session J, 2026-10-22): the same input and
    output layers, initialization and --io_proj_mode, on top of EmbeddingModelDocEncNoProj.

    Queries and documents share self.encoder here, so the output layer must not touch documents. Every
    query-side call (forward, generate) passes inputs_embeds, and every document call (encode_documents,
    and gen_embed_new.py's corpus embedding through model.encoder) passes input_ids. The hook therefore
    projects only calls that came with inputs_embeds. Documents, and the corpus, stay in the encoder's own
    space, as the frozen class keeps them in the frozen encoder's.
    """

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
            self.encoder.register_forward_hook(self._project_query_output, with_kwargs=True)

    def _project_query_output(self, module, args, kwargs, output):
        if kwargs.get("inputs_embeds") is None:  # document call: leave unprojected
            return output
        h = output.last_hidden_state
        output.last_hidden_state = self.out_proj(h.to(self.out_proj.weight.dtype)).to(h.dtype)
        return output

    def _fed_input(self, vec, q_embeds, q_mask):
        if self.io_proj_mode == "out":
            return super()._fed_input(vec, q_embeds, q_mask)
        x = F.normalize(vec.float(), dim=-1).to(self.in_proj.weight.dtype)
        return self.in_proj(x).to(q_embeds.dtype)


class EmbeddingModelFrozenDocEncSingleQueryProj(inbatch.EmbeddingModelFrozenDocEncSingleQuery):
    """Single-query frozen-doc-encoder model with the learned output layer only (Session F, 2026-10-27, P3-single).

    The legacy src/model.py single-query runs still scored output_projection(h); there is nothing fed back, so no
    input layer. Same forward hook as EmbeddingModelFrozenDocEncProj: every self.encoder call (training forward and
    retrieval, where load_retriever returns model.encoder with this hook attached) pools a projected state.
    Documents go through the separate frozen self.doc_encoder. --io_proj_init as in EmbeddingModelFrozenDocEncProj.
    """

    def __init__(self, opt, retriever=None, tokenizer=None):
        super().__init__(opt, retriever, tokenizer)
        assert getattr(opt, "io_proj_mode", "out") == "out", "single-query --io_proj has the output layer only"
        d = self.embedding.weight.size(1)
        self.out_proj = nn.Linear(d, d, bias=False)
        if getattr(opt, "io_proj_init", "identity") == "identity":
            with torch.no_grad():
                self.out_proj.weight.copy_(torch.eye(d))
        self.encoder.register_forward_hook(self._project_query_output)

    def _project_query_output(self, module, args, output):
        h = output.last_hidden_state
        output.last_hidden_state = self.out_proj(h.to(self.out_proj.weight.dtype)).to(h.dtype)
        return output
