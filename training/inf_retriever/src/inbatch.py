import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import transformers
import logging
import torch.distributed as dist

from src import inf_retriever, dist_utils, utils


logger = logging.getLogger(__name__)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, normalize_embeddings=True):
        super().__init__()
        self.temperature = temperature
        self.normalize_embeddings = normalize_embeddings
        self.ce_loss = nn.CrossEntropyLoss(reduce=False)

    def forward(self, outputs, positive_embeddings, negative_embeddings):
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

        var_sizes = dist_utils.get_varsize(all_embeddings)
        start_idx = 0 if dist_utils.get_rank() == 0 else var_sizes[:dist_utils.get_rank()].sum()
        labels = labels + start_idx
        gather_kemb = dist_utils.varsize_gather(all_embeddings)

        similarity = torch.einsum('bd,cd->bc', outputs / self.temperature, gather_kemb)
        loss = self.ce_loss(similarity, labels)
        
        
        # log stats
        iter_stats = {}
        iter_stats["loss"] = (loss.item(), batch_size)

        predicted_idx = torch.argmax(similarity, dim=-1)
        
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(outputs, dim=0).mean().item()
        stdk = torch.std(all_embeddings, dim=0).mean().item()
        iter_stats["accuracy"] = (accuracy, batch_size)
        iter_stats["stdq"] = (stdq, batch_size)
        iter_stats["stdk"] = (stdk, batch_size)
        
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

    def forward(self, outputs, positive_embeddings, negative_embeddings):
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
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

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
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats





class EmbeddingModelDocEncNoProj(nn.Module):
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
        self.loss_fct = ContrastiveLoss(
            temperature=opt.temperature,
        )

    def _load_retriever(self, model_id):
        # cfg = utils.load_hf(transformers.AutoConfig, model_id, trust_remote_code=True)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id, trust_remote_code=True)
        retriever = utils.load_hf(transformers.AutoModel, model_id, trust_remote_code=True)

        # cfg.name_or_path = model_id

        # # Determine model class based on model_id
        # if "inf" in model_id.lower() or "infly" in model_id.lower():
        #     model_class = inf_retriever.INFRetriever

        # if random_init:
        #     retriever = model_class(cfg)
        # else:
        #     if "inf" in model_id.lower() or "infly" in model_id.lower():
        #         pooling = pooling if pooling else "last_token"
        #         retriever = model_class(cfg, pooling=pooling)
        #     else:
        #         retriever = utils.load_hf(model_class, model_id)

        # retriever.config.pooling = pooling
        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder
    
    
    def encode_documents(self, input_document_ids, attention_mask_document):
        """
        Encode raw document tokens using the shared base_causallm (no projection).

        Returns embeddings in hidden_size space.
        """
        batch_size, num_docs, doc_seq_len = input_document_ids.shape
        assert num_docs % 2 == 0, f"num_documents must be even (pos+neg), got {num_docs}"

        flat_ids = input_document_ids.reshape(-1, doc_seq_len)
        flat_mask = attention_mask_document.reshape(-1, doc_seq_len)

        outputs = self.encoder(input_ids=flat_ids, attention_mask=flat_mask, normalize=self.norm_doc)
        # Last-token pooling → stays in hidden_size space, no projection
        doc_embeddings = self.last_token_pool(outputs.last_hidden_state, flat_mask)

        doc_embeddings = doc_embeddings.reshape(batch_size, num_docs, -1)
        num_pos = num_docs // 2
        positive_embeddings = doc_embeddings[:, :num_pos, :]
        negative_embeddings = doc_embeddings[:, num_pos:, :]
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
            q_tokens: (batch_size, 1, seq_len)
            q_mask:   (batch_size, 1, seq_len)
            q_position_ids: (batch_size, 1, seq_len)
            k_tokens: (batch_size, 1 * 2, seq_len)
                      First 1 are positives, rest are negatives.
            k_mask:   (batch_size, 1 * 2, seq_len)
            labels:   optional, per-query-embedding labels (same semantics as InBatch)

        Returns:
            loss, iter_stats
        """
        bsz = q_tokens.size(0)
        nqe = 1
        q_tokens = q_tokens.unsqueeze(1)
        q_mask = q_mask.unsqueeze(1)
        q_position_ids = q_position_ids.unsqueeze(1)
        k_tokens = k_tokens.unsqueeze(1)
        k_mask = k_mask.unsqueeze(1)
        
        
        loss_mask = q_mask.detach().clone()
        
        # Step 1: Encode documents (hidden_size space, no projection)
        positive_embeddings, negative_embeddings = self.encode_documents(
            k_tokens, k_mask
        )
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
        assert (q_mask[:, :, -1] == 1).all(), "Left-padding expected: last position should always be a real token"

        initial_embeds = self.embedding(q_tokens)
        # current_input = initial_embeds[:, :input_start_for_output, :] 
        outputs = self.encoder(inputs_embeds=initial_embeds, attention_mask=q_mask, position_ids=q_position_ids,)

        # last_token_pool handles left-padding: simply take the last position
        selected_outputs_embeddings = self.last_token_pool(outputs.last_hidden_state, q_mask)
            
        # Step 6: Contrastive loss (all in hidden_size space)
        loss, iter_stats = self.loss_fct(selected_outputs_embeddings, positive_embeddings, negative_embeddings)
        
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
        # iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

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
        # iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        # iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        # iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        # return loss, iter_stats
    
    
    
    
    
    