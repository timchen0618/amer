# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import numpy as np
import math
import random
import transformers
import logging
import torch.distributed as dist

from src import contriever, dist_utils, utils

logger = logging.getLogger(__name__)



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
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

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
    


class InBatchSubtraction(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(InBatchSubtraction, self).__init__()

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
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, inpn_tokens, inpn_mask, len_input_negs, labels, stats_prefix="", iter_stats={}, **kwargs):

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
        if inpn_tokens is not None: 
            inpnemb = self.encoder(input_ids=inpn_tokens, attention_mask=inpn_mask, normalize=self.norm_doc)
            assert len(len_input_negs) == bsz
            start = 0
            for i, _len in enumerate(len_input_negs):
                qemb[i] = qemb[i] - torch.sum(inpnemb[start:start+_len], dim=0)
                start += _len

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



class InBatchGRU(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(InBatchGRU, self).__init__()

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
        self.gru = nn.GRU(input_size=768, hidden_size=768, num_layers=1, batch_first=True, bidirectional=False)

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, inpn_tokens, inpn_mask, len_input_negs, labels, stats_prefix="", iter_stats={}, **kwargs):
        
        bsz = len(q_tokens)
        if labels is not None:  # multiple positives
            len_labels = [len(l) for l in labels]
            labels = torch.tensor([l for ls in labels for l in ls], dtype=torch.long, device=q_tokens.device)
            assert labels.dim() == 1
            len_labels = torch.tensor(len_labels, dtype=torch.long, device=q_tokens.device)  
        else:
            labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        org_qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        
        # Only do GRU is inpn_tokens is not none
        if inpn_tokens is not None: 
            inpnemb = self.encoder(input_ids=inpn_tokens, attention_mask=inpn_mask, normalize=self.norm_doc)
            assert len(len_input_negs) == bsz
            start = 0
            # print(len_input_negs)
            hidden_states = []
            for i, _len in enumerate(len_input_negs):
                # print(start, _len, inpnemb.size(), org_qemb.size(), torch.sum(inpnemb[start:start+_len], dim=0).size())
                if _len > 0:
                    _, hn = self.gru(inpnemb[start:start+_len], org_qemb[i].unsqueeze(0))
                    start += _len
                    hidden_states.append(hn[-1].unsqueeze(0))
                else:
                    hidden_states.append(org_qemb[i].unsqueeze(0))
                    # org_qemb[i] = hn[-1]
            newqemb = torch.cat(hidden_states, dim=0)
            assert newqemb.size() == org_qemb.size(), (newqemb.size(), org_qemb.size())
            qemb = newqemb
        else:
            qemb = org_qemb

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
            if dist_utils.get_rank() == 0:
                print('negative_sum', negative_sum, negative_sum.size())
            positives = (scores * target_mask)  # (num_queries, num_docs)
            if dist_utils.get_rank() == 0:
                print('pos', positives.tolist(), positives.size())
            contrastive = -torch.log((positives / (positives + negative_sum)) + negative_mask.long())  # (num_queries, num_docs)
            if dist_utils.get_rank() == 0:
                print('contr', contrastive.tolist(), contrastive.size())
            loss = contrastive.sum(dim=1) / len_labels
            if dist_utils.get_rank() == 0:
                print('loss', loss)
            loss = loss.mean()

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        if labels.size(0) == bsz:
            accuracy = 100 * (predicted_idx == labels).float().mean()
        else:
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
        '''
        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        # Encode query & candidate documents
        org_qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        
        # GRU over inpn_tokens to update query embedding
        # Only do GRU is inpn_tokens is not none
        inpnemb = self.encoder(input_ids=inpn_tokens, attention_mask=inpn_mask, normalize=self.norm_doc)
        assert len(len_input_negs) == bsz
        start = 0

        hidden_states = []
        for i, _len in enumerate(len_input_negs):
            if _len > 0:
                _, hn = self.gru(inpnemb[start:start+_len], org_qemb[i].unsqueeze(0))
            else:
                dummy_input = torch.zeros((1, inpnemb.size(-1)), device=org_qemb.device, dtype=inpnemb.dtype)
                _, hn = self.gru(dummy_input, org_qemb[i].unsqueeze(0))
            hidden_states.append(hn[-1].unsqueeze(0))
            start += _len
        newqemb = torch.cat(hidden_states, dim=0)
        assert newqemb.size() == org_qemb.size(), (newqemb.size(), org_qemb.size())
        qemb = newqemb

        # Gather kemb across GPUs and compute dot-product scores
        gather_fn = dist_utils.gather
        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)
        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
            
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats
        '''
    

class InBatchLinearProjection(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(InBatchLinearProjection, self).__init__()

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
        self.hidden_size = 768
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, inpn_tokens, inpn_mask, len_input_negs, labels=None, stats_prefix="", iter_stats={}, **kwargs):
        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        if inpn_tokens is not None:
            inpnemb = self.encoder(input_ids=inpn_tokens, attention_mask=inpn_mask, normalize=self.norm_doc)
            assert len(len_input_negs) == bsz
            start = 0

            q_doc_embs = []
            for i, _len in enumerate(len_input_negs):
                if _len > 0:
                    doc_emb = inpnemb[start:start+_len].mean(dim=0)
                    q_doc_emb = torch.cat([qemb[i].unsqueeze(0), doc_emb.unsqueeze(0)], dim=1)
                    q_doc_embs.append(self.linear(q_doc_emb))
                    start += _len
                else:
                    # Skip linear projection, use original query (same as GRU approach)
                    q_doc_embs.append(qemb[i].unsqueeze(0))

            newqemb = torch.cat(q_doc_embs, dim=0)
            assert newqemb.size() == qemb.size()
            qemb = newqemb
        else:
            qemb = qemb
        '''
        inpnemb = self.encoder(input_ids=inpn_tokens, attention_mask=inpn_mask, normalize=self.norm_doc)
        assert len(len_input_negs) == bsz
        start = 0

        q_doc_embs = []
        for i, _len in enumerate(len_input_negs):
            if _len>0:
                doc_emb = inpnemb[start:start+_len].mean(dim=0)
                q_doc_emb = torch.cat([qemb[i].unsqueeze(0), doc_emb.unsqueeze(0)], dim=1)
                q_doc_embs.append(self.linear(q_doc_emb))
                start += _len
            else:
                q_doc_embs.append(qemb[i].unsqueeze(0))

        newqemb = torch.cat(q_doc_embs, dim=0)
        assert newqemb.size() == qemb.size()
        qemb = newqemb
        '''
        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)
        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        
        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)
        return loss, iter_stats
    
    

# class InBatchSubtractionLinear(nn.Module):
#     def __init__(self, opt, retriever=None, tokenizer=None):
#         super(InBatchSubtractionLinear, self).__init__()

#         self.opt = opt
#         self.norm_doc = opt.norm_doc
#         self.norm_query = opt.norm_query
#         self.label_smoothing = opt.label_smoothing
#         if retriever is None or tokenizer is None:
#             retriever, tokenizer = self._load_retriever(
#                 opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
#             )
#         self.tokenizer = tokenizer
#         self.encoder = retriever
#         self.weights = torch.nn.Parameter(torch.randn(768, 1))
#         self.bias = torch.nn.Parameter(torch.randn(768, 1))
#         self.sigmoid = torch.nn.Sigmoid()
#         # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
#         # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
#         # https://github.com/pytorch/pytorch/issues/57109
#         torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weights)
#             bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#             torch.nn.init.uniform_(self.bias, -bound, bound)

#     def _load_retriever(self, model_id, pooling, random_init):
#         cfg = utils.load_hf(transformers.AutoConfig, model_id)
#         tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

#         if "xlm" in model_id:
#             model_class = contriever.XLMRetriever
#         else:
#             model_class = contriever.Contriever

#         if random_init:
#             retriever = model_class(cfg)
#         else:
#             retriever = utils.load_hf(model_class, model_id)

#         if "bert-" in model_id:
#             if tokenizer.bos_token_id is None:
#                 tokenizer.bos_token = "[CLS]"
#             if tokenizer.eos_token_id is None:
#                 tokenizer.eos_token = "[SEP]"

#         retriever.config.pooling = pooling

#         return retriever, tokenizer

#     def get_encoder(self):
#         return self.encoder

#     def forward(self, q_tokens, q_mask, k_tokens, k_mask, inpn_tokens, inpn_mask, len_input_negs, stats_prefix="", iter_stats={}, **kwargs):

#         bsz = len(q_tokens)
#         if labels is not None:  # multiple positives
        #     labels = torch.tensor(labels, dtype=torch.long, device=q_tokens.device)
        #     assert labels.dim() == 2
        # else:
        #     labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

#         qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
#         kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
                
#         # Only do Subtraction Linear is inpn_tokens is not none
#         if inpn_tokens is not None: 
#             inpnemb = self.encoder(input_ids=inpn_tokens, attention_mask=inpn_mask, normalize=self.norm_doc)
#             assert len(len_input_negs) == bsz
#             start = 0
#             # print(len_input_negs)
#             for i, _len in enumerate(len_input_negs):
#                 if _len > 0:
#                     # print(start, _len, inpnemb.size(), qemb.size(), torch.sum(inpnemb[start:start+_len], dim=0).size())
#                     # after self.linear -> (_len, 768)
#                     doc_emb = self.sigmoid(self.weights * torch.t(inpnemb[start:start+_len]) + self.bias) # (768, _len)
#                     qemb[i] = qemb[i] - torch.sum(doc_emb, dim=1)
#                     start += _len

#         gather_fn = dist_utils.gather

#         gather_kemb = gather_fn(kemb)

#         labels = labels + dist_utils.get_rank() * len(kemb)

#         scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)

#         loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

#         # log stats
#         if len(stats_prefix) > 0:
#             stats_prefix = stats_prefix + "/"
#         iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

#         predicted_idx = torch.argmax(scores, dim=-1)
#         accuracy = 0
        for i, _len in enumerate(len_labels):  # i -> batch index
            pos_indices = labels[start:start+_len]
            if predicted_idx[i].item() in pos_indices:
                accuracy += 1
        accuracy = 100 * (float(accuracy) / len_labels.size(0))
#         stdq = torch.std(qemb, dim=0).mean().item()
#         stdk = torch.std(kemb, dim=0).mean().item()
#         iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
#         iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
#         iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

#         return loss, iter_stats
    


class InBatchSentenceTransformer(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(InBatchSentenceTransformer, self).__init__()

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
        self.hidden_size = 768
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size)

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, inpn_tokens, inpn_mask, input_negs_inst_mask, labels, stats_prefix="", iter_stats={}, **kwargs):
        bsz = len(q_tokens)
        if labels is not None:  # multiple positives
            len_labels = [len(l) for l in labels]
            labels = torch.tensor([l for ls in labels for l in ls], dtype=torch.long, device=q_tokens.device)
            assert labels.dim() == 1
            len_labels = torch.tensor(len_labels, dtype=torch.long, device=q_tokens.device)  
        else:
            labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        org_qemb = self.encoder(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.encoder(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
                
        # Only do Sentence Transformer is inpn_tokens is not none
        if inpn_tokens is not None: 
            inpnemb = self.encoder(input_ids=inpn_tokens, attention_mask=inpn_mask, normalize=self.norm_doc)
            i = 0
            assert len(input_negs_inst_mask) == bsz
            q_doc_embs = []
            for j, inpn_inst_mask in enumerate(input_negs_inst_mask):
                if inpn_inst_mask:
                    q_doc_embs.append(self.linear(torch.cat([org_qemb[j].unsqueeze(0), inpnemb[i].unsqueeze(0)], dim=1)))
                    i += 1
                else:
                    q_doc_embs.append(org_qemb[j].unsqueeze(0))
            newqemb = torch.cat(q_doc_embs, dim=0)
            assert newqemb.size() == org_qemb.size()
            assert i == len(inpnemb), (i, len(inpnemb))
            qemb = newqemb
            
        else:
            qemb = org_qemb

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
    
    
    
    
    

class IRContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(IRContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, query_embeddings, doc_embeddings, labels):
        """
        Compute the contrastive loss for information retrieval, where each query is matched to multiple relevant documents.
        
        Args:
        query_embeddings: Tensor of shape (num_queries, query_dim)
        doc_embeddings: Tensor of shape (num_docs, doc_dim)
        labels: List of lists where labels[i] contains indices of positive documents for query i
        
        Returns:
        loss: Scalar tensor containing the contrastive loss
        """
        # Normalize embeddings
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=1)
        
        # Compute similarity matrix between queries and documents
        similarity_matrix = torch.matmul(query_embeddings, doc_embeddings.T) / self.temperature
        similarity_matrix = torch.exp(similarity_matrix)
        
        # Construct target labels as a binary mask
        num_queries, num_docs = similarity_matrix.shape
        target_mask = torch.zeros((num_queries, num_docs), device=similarity_matrix.device)
        negative_mask = torch.ones((num_queries, num_docs), device=similarity_matrix.device)
        for i, pos_indices in enumerate(labels):
            target_mask[i, pos_indices] = 1  # Assign positive labels
            negative_mask[i, pos_indices] = 0
        negative_sum = (similarity_matrix * negative_mask).sum(dim=1, keepdim=True)  # (num_queries, 1)
        len_labels = torch.tensor([len(pos_indices) for pos_indices in labels], device=similarity_matrix.device) # (num_queries, )
        positives = (similarity_matrix * target_mask)  # (num_queries, num_docs)
        contrastive = -torch.log(positives / (positives + negative_sum))  # (num_queries, num_docs)
        loss = contrastive.sum(dim=1) / len_labels
        loss = loss.mean()
        
        return loss
