import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import transformers
import logging
import torch.distributed as dist

from training.inf_retriever.src import dist_utils, utils


logger = logging.getLogger(__name__)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, normalize_embeddings=True):
        super().__init__()
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, outputs, positive_embeddings, negative_embeddings, stats_prefix="", iter_stats={}):
        if self.normalize_embeddings:
            outputs = F.normalize(outputs, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)

        batch_size, k, d = outputs.shape
        labels = torch.arange(batch_size * k).long().to(outputs.device)
        outputs = outputs.view(batch_size * k, d)
        positive_embeddings = positive_embeddings.view(batch_size * k, d)
        negative_embeddings = negative_embeddings.view(batch_size * k, d)
        all_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)

        # handle distributed training: Use var size gather to handle different sizes of embeddings
        var_sizes = dist_utils.get_varsize(all_embeddings)
        start_idx = 0 if dist_utils.get_rank() == 0 else var_sizes[:dist_utils.get_rank()].sum()
        labels = labels + start_idx
        gather_kemb = dist_utils.varsize_gather(all_embeddings)
        
        # original gather
        # gather_kemb = dist_utils.gather(all_embeddings)
        # labels = labels + dist_utils.get_rank() * len(all_embeddings)

        similarity = torch.einsum('bd,cd->bc', outputs / self.temperature, gather_kemb)
        loss = self.ce_loss(similarity, labels)
        
        
        # log stats
        predicted_idx = torch.argmax(similarity, dim=-1)
        
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(outputs, dim=0).mean().item()
        stdk = torch.std(all_embeddings, dim=0).mean().item()
        iter_stats[f"{stats_prefix}/accuracy"] = (accuracy, batch_size)
        iter_stats[f"{stats_prefix}/stdq"] = (stdq, batch_size)
        iter_stats[f"{stats_prefix}/stdk"] = (stdk, batch_size)
        
        iter_stats[f"{stats_prefix}/loss"] = (loss.mean().item(), batch_size)
        return loss.mean(), iter_stats


class HungarianContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, use_eos=False, normalize_embeddings=True):
        super().__init__()
        from scipy.optimize import linear_sum_assignment
        self.linear_sum_assignment = linear_sum_assignment
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.use_eos = use_eos
        self.normalize_embeddings = normalize_embeddings

    def forward(self, outputs, positive_embeddings, negative_embeddings, stats_prefix="", iter_stats=None):
        if iter_stats is None:
            iter_stats = {}
        loss = self._core_forward(outputs, positive_embeddings, negative_embeddings)
        iter_stats[f"{stats_prefix}/loss"] = (loss.item(), outputs.shape[0])
        return loss, iter_stats

    def _core_forward(self, outputs, positive_embeddings, negative_embeddings):
        if self.normalize_embeddings:
            outputs = F.normalize(outputs, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)

        batch_size, k, d = outputs.shape
        outputs = outputs.view(batch_size * k, d)
        positive_embeddings = positive_embeddings.view(batch_size * k, d)
        negative_embeddings = negative_embeddings.view(batch_size * k, d)
        all_embeddings = torch.cat([positive_embeddings, negative_embeddings], dim=0)
        var_sizes = dist_utils.get_varsize(all_embeddings)

        gather_kemb = dist_utils.varsize_gather(all_embeddings)

        similarity = torch.einsum('bd,cd->bc', outputs / self.temperature, gather_kemb)
        similarity = self.log_softmax(similarity)
        losses = []
        for i in range(batch_size):
            start_idx = 0 if dist_utils.get_rank() == 0 else var_sizes[:dist_utils.get_rank()].sum()
            start_idx += k * i
            batch_scores = similarity[k * i:k * (i + 1)]
            cost_matrix = batch_scores[:, start_idx:start_idx + k]

            row_ind, col_ind = self.linear_sum_assignment(cost_matrix.detach().cpu().numpy(), maximize=True)
            costs = cost_matrix[row_ind, col_ind]
            costs = -(costs)
            losses.append(costs.mean())
        return torch.stack(losses).mean()


class HungarianMaskedContrastiveLoss(nn.Module):
    """Multi-positive contrastive loss with Hungarian assignment and same-example
    positive masking.

    Given ``outputs`` of shape (bsz, k, d) — k auto-regressive query embeddings
    per example — and ``positive_embeddings`` / ``negative_embeddings`` each of
    shape (bsz, k, d) (the j-th row is the j-th gold / random negative for the
    example), this loss:

    1. Uses the scipy linear-sum-assignment to match each of the k predicted
       embeddings to the best-scoring positive *of the same example*. This
       removes the arbitrary "embedding j must match gold j" supervision.
    2. When computing the softmax denominator, masks out the *other* positives
       of the same example so they do not count as in-batch negatives. Only
       the Hungarian-assigned positive remains among the same-example
       candidates; every negative (local + cross-rank) and every positive /
       negative from *other* examples stays in the pool as usual.

    Differences from ``HungarianContrastiveLoss``:
        - HungarianContrastiveLoss takes log_softmax over the full candidate
          pool *before* the Hungarian step, so the other k-1 positives of the
          same example are inside the log-partition and push the loss up even
          when the model has ranked them correctly (false negatives).
        - HungarianContrastiveLoss has no stats / iter_stats plumbing and
          returns only the scalar loss, so it cannot be dropped into
          EmbeddingModelDocEncNoProj without changes.
        - This loss reports both a strict accuracy (argmax == Hungarian
          assignment, over the masked logits) and a relaxed accuracy (argmax
          over the *unmasked* similarity is any same-example positive), which
          is the fair counterpart to the standard_org_q accuracy curve.
    """

    def __init__(self, temperature=0.05, normalize_embeddings=True):
        super().__init__()
        from scipy.optimize import linear_sum_assignment
        self.linear_sum_assignment = linear_sum_assignment
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings

    def forward(self, outputs, positive_embeddings, negative_embeddings, stats_prefix="", iter_stats={}):
        if self.normalize_embeddings:
            outputs = F.normalize(outputs, dim=-1)
            positive_embeddings = F.normalize(positive_embeddings, dim=-1)
            negative_embeddings = F.normalize(negative_embeddings, dim=-1)

        batch_size, k, d = outputs.shape
        bk = batch_size * k
        outputs_flat = outputs.reshape(bk, d)
        positive_flat = positive_embeddings.reshape(bk, d)
        negative_flat = negative_embeddings.reshape(bk, d)

        # Order within each rank's gather chunk: [bsz*k positives ; bsz*k negatives]
        all_embeddings = torch.cat([positive_flat, negative_flat], dim=0)
        var_sizes = dist_utils.get_varsize(all_embeddings)
        rank = dist_utils.get_rank()
        start_idx_local = 0 if rank == 0 else int(var_sizes[:rank].sum().item())
        gather_kemb = dist_utils.varsize_gather(all_embeddings)

        # Local positives live at [local_pos_start, local_pos_end) in the gathered tensor.
        # Positives from other ranks belong to different examples, so they are legitimate
        # in-batch negatives and need no masking.
        local_pos_start = start_idx_local
        local_pos_end = start_idx_local + bk

        similarity = torch.einsum('bd,cd->bc', outputs_flat / self.temperature, gather_kemb)  # (bk, total)

        # Hungarian assignment per example, using the (k, k) block of similarity
        # between this example's k predicted embeddings and its k local positives.
        assigned_targets = torch.empty(bk, dtype=torch.long, device=outputs.device)
        sim_detached = similarity.detach().float().cpu().numpy()
        for i in range(batch_size):
            pos_start = local_pos_start + k * i
            cost = sim_detached[k * i : k * (i + 1), pos_start : pos_start + k]  # (k, k)
            row_ind, col_ind = self.linear_sum_assignment(cost, maximize=True)
            for ri, ci in zip(row_ind, col_ind):
                assigned_targets[k * i + ri] = pos_start + ci

        # Mask out same-example positives except the Hungarian-assigned target.
        q_example = torch.arange(bk, device=outputs.device) // k   # (bk,)
        cand_example = torch.arange(bk, device=outputs.device) // k  # example of each local positive slot
        same_example = q_example.unsqueeze(1).eq(cand_example.unsqueeze(0))  # (bk, bk)

        mask = torch.ones_like(similarity, dtype=torch.bool)          # True = keep
        mask[:, local_pos_start:local_pos_end] = ~same_example        # drop all same-example local positives
        mask[torch.arange(bk, device=outputs.device), assigned_targets] = True  # re-enable the assigned one

        masked_similarity = similarity.masked_fill(~mask, float('-inf'))
        loss = F.cross_entropy(masked_similarity, assigned_targets)

        with torch.no_grad():
            strict_correct = (masked_similarity.argmax(dim=-1) == assigned_targets)
            strict_acc = 100.0 * strict_correct.float().mean()

            raw_pred = similarity.argmax(dim=-1)                      # (bk,)
            # argmax falls inside this example's local positive block?
            pos_block_start = local_pos_start + k * q_example
            pos_block_end = pos_block_start + k
            relaxed_correct = (raw_pred >= pos_block_start) & (raw_pred < pos_block_end)
            relaxed_acc = 100.0 * relaxed_correct.float().mean()

            stdq = torch.std(outputs_flat, dim=0).mean().item()
            stdk = torch.std(all_embeddings, dim=0).mean().item()

        iter_stats[f"{stats_prefix}/accuracy"] = (relaxed_acc.item(), batch_size)
        iter_stats[f"{stats_prefix}/accuracy_strict"] = (strict_acc.item(), batch_size)
        iter_stats[f"{stats_prefix}/stdq"] = (stdq, batch_size)
        iter_stats[f"{stats_prefix}/stdk"] = (stdk, batch_size)
        iter_stats[f"{stats_prefix}/loss"] = (loss.item(), batch_size)
        return loss, iter_stats


LOSS_REGISTRY = {
    "contrastive": ContrastiveLoss,
    "hungarian_masked": HungarianMaskedContrastiveLoss,
    "hungarian": HungarianContrastiveLoss,
}


def build_loss(opt):
    """Instantiate the loss module requested by ``opt.loss_fn``.

    ``loss_fn='auto'`` resolves to ``hungarian_masked`` for multi mode and
    ``contrastive`` otherwise, preserving the old default behaviour.
    """
    name = getattr(opt, "loss_fn", "auto")
    if name == "auto":
        name = "hungarian_masked" if getattr(opt, "training_mode", None) == "multi" else "contrastive"
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss_fn={name!r}; valid choices: {sorted(LOSS_REGISTRY)}")
    print(f"[build_loss] training_mode={getattr(opt, 'training_mode', None)!r} -> loss_fn={name!r}", flush=True)
    return LOSS_REGISTRY[name](temperature=opt.temperature)


class InBatch(nn.Module):
    
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(InBatch, self).__init__()
        
        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
    
    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id, trust_remote_code=True)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id, trust_remote_code=True)

        cfg.name_or_path = model_id

        # Determine model class based on model_id
        if "inf" in model_id.lower() or "infly" in model_id.lower():
            model_class = inf_retriever.INFRetriever
        
        if random_init:
            retriever = model_class(cfg)
        else:
            if "inf" in model_id.lower() or "infly" in model_id.lower():
                pooling = pooling if pooling else "last_token"
                retriever = model_class(cfg, pooling=pooling)
            else:
                retriever = utils.load_hf(model_class, model_id)

        # Set up special tokens for different models
        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"
        elif "inf" in model_id.lower() or "infly" in model_id.lower():
            pass

        retriever.config.pooling = pooling
        return retriever, tokenizer
    
    def get_encoder(self):
        return self.encoder
    
    def forward(self, q_tokens, q_mask, k_tokens, k_mask, labels=None, stats_prefix="", iter_stats={}, **kwargs):

        bsz = len(q_tokens)
        if labels is not None:  # multiple positives
            len_labels = [len(l) for l in labels]
            labels = torch.tensor([l for ls in labels for l in ls], dtype=torch.long, device=q_tokens.device)
            assert labels.dim() == 1
            len_labels = torch.tensor(len_labels, dtype=torch.long, device=q_tokens.device)            
        else:
            labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)
        
        if labels.size(0) == bsz:
            scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)
            loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)
        else:
            scores = torch.exp(torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb))
            num_queries, num_docs = scores.shape
            target_mask = torch.zeros((num_queries, num_docs), device=scores.device)
            negative_mask = torch.ones((num_queries, num_docs), device=scores.device)
            start = 0
            for i, _len in enumerate(len_labels):  # i -> batch index
                pos_indices = labels[start:start+_len]
                target_mask[i, pos_indices] = 1   # Assign positive labels
                negative_mask[i, pos_indices] = 0 # Assign negative labels
                start += _len
            negative_sum = (scores * negative_mask).sum(dim=1, keepdim=True)  # (num_queries, 1)
            positives = (scores * target_mask)  # (num_queries, num_docs)
            contrastive = -torch.log(positives / (positives + negative_sum))  # (num_queries, num_docs)
            loss = contrastive.sum(dim=1) / len_labels
            loss = loss.mean()

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}/loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        
        if labels.size(0) == bsz:
            accuracy = 100 * (predicted_idx == labels).float().mean()
        else:
            start = 0
            accuracy = 0
            for i, _len in enumerate(len_labels):  # i -> batch index
                pos_indices = labels[start:start+_len]
                if predicted_idx[i].item() in pos_indices:
                    accuracy += 1
            accuracy = 100 * (float(accuracy) / len_labels.size(0))
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}/accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}/stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}/stdk"] = (stdk, bsz)

        return loss, iter_stats





class EmbeddingModelDocEncNoProj(nn.Module):
    """
    Autoregressive multi-query encoder: one left-padded query sequence produces
    num_query_embeddings query vectors (last-token hidden states), with hidden
    states fed back as additional input tokens (no projection).

    Forward:
        q_tokens, q_mask, q_position_ids: (batch_size, seq_len) — left-padded query
        k_tokens, k_mask: (batch_size, num_query_embeddings * 2, seq_len)
            First num_query_embeddings slots are positive docs, second half negatives.

    Query output shape passed to ContrastiveLoss: (batch_size, num_query_embeddings, dim).
    """

    def __init__(self, opt, retriever=None, tokenizer=None):
        super(EmbeddingModelDocEncNoProj, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing

        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
        self.embedding = retriever.get_input_embeddings()
        # Loss module is selected via --loss_fn (with 'auto' -> hungarian_masked
        # for multi mode, contrastive otherwise). See build_loss above.
        self.loss_fct = build_loss(opt)

    def _load_retriever(self, model_id):
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        retriever = transformers.AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        if hasattr(retriever, 'gradient_checkpointing_enable'):
            retriever.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("grad checkpoint enabled for EmbeddingModelDocEncNoProj")
        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder
    
    
    def encode_documents(self, input_document_ids, attention_mask_document):
        """
        Encode document token batches in hidden_size space (no projection).

        - 3D input (batch_size, 2 * nqe, seq_len): returns positives and negatives
          each (batch_size, nqe, hidden_size).
        - 2D input (N, seq_len) with N even (e.g. concat of all golds then all negs):
          returns (N/2, hidden_size) for each half — used by evaluate().
          The first half is all golds, the second half is all negatives.
        """
        # if input_document_ids.dim() == 3:
        #     bsz, num_docs, doc_seq_len = input_document_ids.shape
        #     assert num_docs % 2 == 0, f"num_documents must be even (pos+neg), got {num_docs}"
        #     nqe = num_docs // 2
        #     flat_ids = input_document_ids.reshape(-1, doc_seq_len)
        #     flat_mask = attention_mask_document.reshape(-1, doc_seq_len)
        #     outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        #     doc_embeddings = self.last_token_pool(outputs.last_hidden_state, flat_mask)
        #     half = flat_ids.size(0) // 2
        #     positive_embeddings = doc_embeddings[:half, :].view(bsz, nqe, -1)
        #     negative_embeddings = doc_embeddings[half:, :].view(bsz, nqe, -1)
        #     return positive_embeddings, negative_embeddings

        flat_ids = input_document_ids
        flat_mask = attention_mask_document
        batch_size = flat_ids.size(0) // 2
        assert flat_ids.size(0) % 2 == 0, f"num_documents must be even (pos+neg), got {flat_ids.size(0)}"

        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        doc_embeddings = self.last_token_pool(outputs.last_hidden_state, flat_mask)

        positive_embeddings = doc_embeddings[:batch_size, :]
        negative_embeddings = doc_embeddings[batch_size:, :]
        assert positive_embeddings.size(0) == batch_size
        assert negative_embeddings.size(0) == batch_size
        return positive_embeddings, negative_embeddings

    def last_token_pool(self, last_hidden_states, attention_mask):
        """
        Pool the last token as per INF-Retriever documentation
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        


    @torch.no_grad()
    def generate(self, q_tokens, q_mask, q_position_ids, max_new_tokens=5):
        """Autoregressively generate multiple query embeddings at inference time.

        Each step pools the last-token hidden state as one query embedding, then feeds
        it back as the next input token. No teacher forcing or scheduled sampling.

        Args:
            q_tokens:       (batch_size, seq_len) — left-padded query token ids
            q_mask:         (batch_size, seq_len) — attention mask (1 = real, 0 = pad)
            q_position_ids: (batch_size, seq_len) — position ids for left-padded input
            max_new_tokens: number of query embeddings to generate

        Returns:
            (batch_size, max_new_tokens, hidden_size) float tensor, unnormalized
        """
        bsz = q_tokens.size(0)
        device = q_tokens.device

        assert (q_mask[:, -1] == 1).all(), \
            "Left-padding expected: last token position must always be a real token"

        current_input = self.embedding(q_tokens)
        attn = q_mask
        pos = q_position_ids
        all_outputs = []

        for j in range(max_new_tokens):
            outputs = self.encoder(
                inputs_embeds=current_input,
                attention_mask=attn,
                position_ids=pos,
            )
            next_hidden = self.last_token_pool(outputs.last_hidden_state, attn)
            all_outputs.append(next_hidden.unsqueeze(1))

            if j == max_new_tokens - 1:
                break

            next_tok = next_hidden.unsqueeze(1)
            current_input = torch.cat((current_input, next_tok), dim=1)
            attn = torch.cat(
                (attn, torch.ones(bsz, 1, dtype=attn.dtype, device=device)),
                dim=1,
            )
            pos = torch.cat((pos, pos[:, -1:] + 1), dim=1)

        return torch.cat(all_outputs, dim=1)  # (bsz, max_new_tokens, hidden_size)

    def forward(self, q_tokens, q_mask, q_position_ids, k_tokens, k_mask, labels=None, stats_prefix="", iter_stats={}, **kwargs):
        """
        Args:
            q_tokens: (batch_size, seq_len)
            q_mask:   (batch_size, seq_len)
            q_position_ids: (batch_size, seq_len)
            k_tokens: (batch_size*num_query_embeddings * 2, seq_len)
            k_mask:   (batch_size*num_query_embeddings * 2, seq_len)
            labels:   reserved (optional)
            sampling_rate (kwargs): probability of feeding back the predicted hidden state
                on non-final steps; else teacher (positive doc) embedding. Default 1.0.

        Returns:
            loss, iter_stats
        """
        assert k_tokens.dim() == 2 and k_mask.dim() == 2
        assert k_tokens.size(0) % 2 == 0
        bsz = q_tokens.size(0)
        # assert k_tokens.size(0) == bsz
        
        sampling_rate = kwargs.get("sampling_rate", 1.0)

        positive_embeddings, negative_embeddings = self.encode_documents(k_tokens, k_mask)
        teacher_embeddings = positive_embeddings.reshape(bsz, -1, positive_embeddings.size(-1))
        negative_embeddings = negative_embeddings.reshape(bsz, -1, negative_embeddings.size(-1))
        output_len = teacher_embeddings.size(1)

        assert (q_mask[:, -1] == 1).all(), (
            "Left-padding expected: last position should always be a real token",
            q_mask[:, -1],
        )

        current_input = self.embedding(q_tokens)
        attn = q_mask
        pos = q_position_ids
        device = q_tokens.device
        all_outputs = []

        for j in range(output_len):
            outputs = self.encoder(
                inputs_embeds=current_input,
                attention_mask=attn,
                position_ids=pos,
            )
            next_hidden = self.last_token_pool(outputs.last_hidden_state, attn)
            all_outputs.append(next_hidden.unsqueeze(1))

            if j == output_len - 1:
                break

            use_predicted = (torch.rand(bsz, 1, 1, device=device) < sampling_rate)
            predicted = next_hidden.unsqueeze(1)
            teacher = teacher_embeddings[:, j, :].unsqueeze(1).to(dtype=next_hidden.dtype)
            next_tok = torch.where(use_predicted, predicted, teacher)

            current_input = torch.cat((current_input, next_tok), dim=1)
            attn = torch.cat(
                (attn, torch.ones(bsz, 1, dtype=attn.dtype, device=device)),
                dim=1,
            )
            pos = torch.cat((pos, pos[:, -1:] + 1), dim=1)

        selected_outputs_embeddings = torch.cat(all_outputs, dim=1)
        loss, iter_stats = self.loss_fct(
            selected_outputs_embeddings,
            positive_embeddings,
            negative_embeddings,
            stats_prefix=stats_prefix,
            iter_stats=iter_stats,
        )
        return loss, iter_stats



    # def forward(self, q_tokens, q_mask, q_position_ids, k_tokens, k_mask, labels=None, stats_prefix="", iter_stats={}, **kwargs):
    #     """
    #     Args:
    #         q_tokens: (batch_size, num_query_embeddings, seq_len)
    #         q_mask:   (batch_size, num_query_embeddings, seq_len)
    #         k_tokens: (batch_size, num_query_embeddings * 2, seq_len)
    #                   First num_query_embeddings are positives, rest are negatives.
    #         k_mask:   (batch_size, num_query_embeddings * 2, seq_len)
    #         labels:   optional, per-query-embedding labels (same semantics as InBatch)

    #     Returns:
    #         loss, iter_stats
    #     """
    #     bsz = q_tokens.size(0)
    #     nqe = self.num_query_embeddings
        
    #     loss_mask = q_mask.detach().clone()
        
    #     # Step 1: Encode documents (hidden_size space, no projection)
    #     positive_embeddings, negative_embeddings = self.encode_documents(
    #         k_tokens, k_mask
    #     )
    #     labels = positive_embeddings  # teacher forcing targets (hidden_size)
    #     output_len = labels.size(1)
        
    #     # Step 2: Run causal LM on query tokens
    #     input_start_for_output = q_mask.sum(dim=1).max()
    #     assert q_mask[:, input_start_for_output:].sum().item() == 0

    #     initial_embeds = self.embedding(q_tokens)
    #     current_input = initial_embeds[:, :input_start_for_output, :]
    #     outputs = self.encoder(inputs_embeds=initial_embeds, attention_mask=q_mask, position_ids=q_position_ids,)
        
    #     # Step 3: Autoregressive generation with scheduled sampling
    #     # No projection — hidden states are fed back directly
    #     sampling_rate = inputs['sampling_rate']
    #     all_outputs = []
    #     for j in range(output_len):
    #         outputs = self.encoder(
    #             inputs_embeds=current_input,
    #             position_ids=q_position_ids[:, :input_start_for_output + j],
    #             attention_mask=q_mask[:, :input_start_for_output + j],
    #         )

    #         next_emb = outputs.last_hidden_state[:, input_start_for_output + j - 1, :]
    #         all_outputs.append(next_emb.unsqueeze(1))

    #         # Scheduled sampling: predicted hidden state or teacher (doc hidden state)
    #         use_predicted = (torch.rand(current_input.size(0), 1, 1) < sampling_rate).to(current_input.device)
    #         predicted = next_emb.unsqueeze(1)                   # already in hidden_size
    #         teacher = labels[:, j].float().unsqueeze(1)         # also in hidden_size
    #         next_input = torch.where(use_predicted, predicted, teacher)

    #         current_input = torch.cat((current_input, next_input), dim=1)

    #         inputs['position_ids'][:, input_start_for_output + j:] = inputs['position_ids'][:, input_start_for_output + j:] + 1
    #         inputs['attention_mask'][:, input_start_for_output + j] = 1
        
        
        
        
        
        
        

        # # Flatten queries: (batch_size * nqe, seq_len)
        # q_tokens_flat = q_tokens.view(bsz * nqe, -1)
        # q_mask_flat = q_mask.view(bsz * nqe, -1)

        # # Flatten documents: (batch_size * nqe * 2, seq_len)
        # k_tokens_flat = k_tokens.view(bsz * nqe * 2, -1)
        # k_mask_flat = k_mask.view(bsz * nqe * 2, -1)


        










        # # Encode queries and documents with the shared encoder
        # qemb = self.encoder(input_ids=q_tokens_flat, attention_mask=q_mask_flat, normalize=self.norm_query)
        # kemb = self.encoder(input_ids=k_tokens_flat, attention_mask=k_mask_flat, normalize=self.norm_doc)

        # # kemb: (bsz * nqe * 2, dim) — split into positives and negatives
        # kemb_all = kemb.view(bsz, nqe * 2, -1)
        # pos_kemb = kemb_all[:, :nqe, :].reshape(bsz * nqe, -1)   # (bsz * nqe, dim)
        # neg_kemb = kemb_all[:, nqe:, :].reshape(bsz * nqe, -1)   # (bsz * nqe, dim)

        # # Effective batch size: each query embedding is treated as an independent query
        # effective_bsz = bsz * nqe

        # # Gather positive doc embeddings across processes for in-batch negatives
        # gather_fn = dist_utils.gather
        # gather_pos_kemb = gather_fn(pos_kemb)   # (world_size * bsz * nqe, dim)
        # gather_neg_kemb = gather_fn(neg_kemb)   # (world_size * bsz * nqe, dim)

        # # Candidate pool: in-batch positives (used as negatives for other queries) + explicit negatives
        # # gather_kemb: (world_size * bsz * nqe * 2, dim)
        # gather_kemb = torch.cat([gather_pos_kemb, gather_neg_kemb], dim=0)

        # if labels is not None:
        #     len_labels = [len(l) for l in labels]
        #     labels = torch.tensor([l for ls in labels for l in ls], dtype=torch.long, device=q_tokens.device)
        #     assert labels.dim() == 1
        #     len_labels = torch.tensor(len_labels, dtype=torch.long, device=q_tokens.device)
        # else:
        #     # Each query embedding's positive is at the corresponding index in gather_pos_kemb
        #     labels = torch.arange(0, effective_bsz, dtype=torch.long, device=q_tokens.device)

        # labels = labels + dist_utils.get_rank() * effective_bsz

        # if labels.size(0) == effective_bsz:
        #     scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)
        #     loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)
        # else:
        #     scores = torch.exp(torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb))
        #     num_queries, num_docs = scores.shape
        #     target_mask = torch.zeros((num_queries, num_docs), device=scores.device)
        #     negative_mask = torch.ones((num_queries, num_docs), device=scores.device)
        #     start = 0
        #     for i, _len in enumerate(len_labels):
        #         pos_indices = labels[start:start+_len]
        #         target_mask[i, pos_indices] = 1
        #         negative_mask[i, pos_indices] = 0
        #         start += _len
        #     negative_sum = (scores * negative_mask).sum(dim=1, keepdim=True)
        #     positives = (scores * target_mask)
        #     contrastive = -torch.log(positives / (positives + negative_sum))
        #     loss = contrastive.sum(dim=1) / len_labels
        #     loss = loss.mean()

        # # log stats
        # if len(stats_prefix) > 0:
        #     stats_prefix = stats_prefix + "/"
        # iter_stats[f"{stats_prefix}/loss"] = (loss.item(), bsz)

        # predicted_idx = torch.argmax(scores, dim=-1)

        # if labels.size(0) == effective_bsz:
        #     accuracy = 100 * (predicted_idx == labels).float().mean()
        # else:
        #     start = 0
        #     accuracy = 0
        #     for i, _len in enumerate(len_labels):
        #         pos_indices = labels[start:start+_len]
        #         if predicted_idx[i].item() in pos_indices:
        #             accuracy += 1
        #     accuracy = 100 * (float(accuracy) / len_labels.size(0))
        # stdq = torch.std(qemb, dim=0).mean().item()
        # stdk = torch.std(kemb, dim=0).mean().item()
        # iter_stats[f"{stats_prefix}/accuracy"] = (accuracy, bsz)
        # iter_stats[f"{stats_prefix}/stdq"] = (stdq, bsz)
        # iter_stats[f"{stats_prefix}/stdk"] = (stdk, bsz)

        # return loss, iter_stats
    

class EmbeddingModelDocEncNoProjSingleQuery(nn.Module):
    """
    Multi-query variant of InBatch. Uses the same shared encoder for queries and
    documents, but each query produces num_query_embeddings embeddings.

    Query output shape:  (batch_size, num_query_embeddings, embedding_dim)
    Document input:      num_query_embeddings positive + num_query_embeddings negative
                         docs per sample, encoded with the same encoder.

    Forward signature mirrors InBatch:
        q_tokens:  (batch_size, num_query_embeddings, seq_len)
        k_tokens:  (batch_size, num_query_embeddings * 2, seq_len)
                   first num_query_embeddings are positives, rest are negatives
    """

    def __init__(self, opt, retriever=None, tokenizer=None):
        super(EmbeddingModelDocEncNoProjSingleQuery, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing

        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
        self.embedding = retriever.get_input_embeddings()
        self.loss_fct = ContrastiveLoss(
            temperature=opt.temperature,
        )

    def _load_retriever(self, model_id):
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        retriever = transformers.AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False,
        )
        if hasattr(retriever, 'gradient_checkpointing_enable'):
            retriever.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            print("grad checkpoint enabled for EmbeddingModelDocEncNoProj")
        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder
    
    
    def encode_documents(self, input_document_ids, attention_mask_document):
        """
        Encode raw document tokens using the shared base_causallm (no projection).

        Returns embeddings in hidden_size space.
        """
        flat_ids = input_document_ids
        flat_mask = attention_mask_document
        batch_size = flat_ids.size(0) // 2
        assert flat_ids.size(0) % 2 == 0, f"num_documents must be even (pos+neg), got {flat_ids.size(0)}"

        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        # Last-token pooling → stays in hidden_size space, no projection
        doc_embeddings = self.last_token_pool(outputs.last_hidden_state, flat_mask)

        positive_embeddings = doc_embeddings[:batch_size, :] # first half of the embeddings are positive
        negative_embeddings = doc_embeddings[batch_size:, :] # second half of the embeddings are negative
        assert positive_embeddings.size(0) == batch_size
        assert negative_embeddings.size(0) == batch_size
        return positive_embeddings, negative_embeddings

    def last_token_pool(self, last_hidden_states, attention_mask):
        """
        Pool the last token as per INF-Retriever documentation
        """
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
        


    def forward(self, q_tokens, q_mask, q_position_ids, k_tokens, k_mask, labels=None, stats_prefix="", iter_stats={}, **kwargs):
        """
        Args:
            q_tokens: (batch_size, seq_len)
            q_mask:   (batch_size, seq_len)
            q_position_ids: (batch_size, seq_len)
            k_tokens: (batch_size, 1 * 2, seq_len)
                      First 1 are positives, rest are negatives.
            k_mask:   (batch_size, 1 * 2, seq_len)
            labels:   optional, per-query-embedding labels (same semantics as InBatch)

        Returns:
            loss, iter_stats
        """
        bsz = q_tokens.size(0)
        nqe = 1
        
        
        loss_mask = q_mask.detach().clone()
        
        # Step 1: Encode documents (hidden_size space, no projection)
        positive_embeddings, negative_embeddings = self.encode_documents(k_tokens, k_mask)
        labels = positive_embeddings  # teacher forcing targets (hidden_size)
        output_len = labels.size(1)
        
        # Step 2: Run causal LM on query tokens (left-padded)
        # With left-padding, padding is at the start; real tokens are contiguous at the end.
        # Verify: for each sample, leading positions are padding (0) and trailing are real (1).
        
        # input_start_for_output = q_mask.sum(dim=1).max()
        # assert q_mask[:, input_start_for_output:].sum().item() == 0
        seq_len = q_mask.size(-1)
        num_real = q_mask.sum(dim=-1)  # (bsz, 1)
        pad_len = seq_len - num_real   # (bsz, 1) — number of leading pad tokens per sample
        assert (q_mask[:, -1] == 1).all(), ("Left-padding expected: last position should always be a real token", q_mask[:, -1])

        initial_embeds = self.embedding(q_tokens)
        print('initial_embeds', initial_embeds.shape)
        # current_input = initial_embeds[:, :input_start_for_output, :] 
        outputs = self.encoder(inputs_embeds=initial_embeds, attention_mask=q_mask, position_ids=q_position_ids,)
        print('outputs.last_hidden_state', outputs.last_hidden_state.shape)
        # last_token_pool handles left-padding: simply take the last position
        selected_outputs_embeddings = self.last_token_pool(outputs.last_hidden_state, q_mask)
        print('selected_outputs_embeddings', selected_outputs_embeddings.shape)
        selected_outputs_embeddings = selected_outputs_embeddings.unsqueeze(1)
        # Step 6: Contrastive loss (all in hidden_size space)
        loss, iter_stats = self.loss_fct(selected_outputs_embeddings, positive_embeddings, negative_embeddings, stats_prefix=stats_prefix, iter_stats=iter_stats)
        print('loss', loss.shape, loss)
        return loss, iter_stats