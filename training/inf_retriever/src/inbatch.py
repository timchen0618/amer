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



class InBatchINFGRU(nn.Module):
    
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(InBatchINFGRU, self).__init__()
        
        self.opt = opt
        self.norm_doc = opt.norm_doc if hasattr(opt, 'norm_doc') else True
        self.norm_query = opt.norm_query if hasattr(opt, 'norm_query') else True
        self.label_smoothing = opt.label_smoothing if hasattr(opt, 'label_smoothing') else 0.0
        
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id,
                pooling=opt.pooling if hasattr(opt, 'pooling') else "last_token",
                random_init=opt.random_init if hasattr(opt, 'random_init') else False
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
        
        # INF-Retriever uses 1536 dimensions
        hidden_size = 1536
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
    
    def _load_retriever(self, model_id, pooling, random_init):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        
        base_model = transformers.AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        retriever = inf_retriever.INFRetriever(base_model, pooling=pooling)
        
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
        
        # Only do GRU if inpn_tokens is not none
        if inpn_tokens is not None: 
            inpnemb = self.encoder(input_ids=inpn_tokens, attention_mask=inpn_mask, normalize=self.norm_doc)
            assert len(len_input_negs) == bsz
            start = 0
            hidden_states = []
            for i, _len in enumerate(len_input_negs):
                if _len > 0:
                    _, hn = self.gru(inpnemb[start:start+_len], org_qemb[i].unsqueeze(0))
                    start += _len
                    hidden_states.append(hn[-1].unsqueeze(0))
                else:
                    hidden_states.append(org_qemb[i].unsqueeze(0))
            newqemb = torch.cat(hidden_states, dim=0)
            assert newqemb.size() == org_qemb.size()
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
            for i, _len in enumerate(len_labels):
                pos_indices = labels[start:start+_len]
                target_mask[i, pos_indices] = 1
                negative_mask[i, pos_indices] = 0
                start += _len
            negative_sum = (scores * negative_mask).sum(dim=1, keepdim=True)
            positives = (scores * target_mask)
            contrastive = -torch.log((positives / (positives + negative_sum)) + negative_mask.long())
            loss = contrastive.sum(dim=1) / len_labels
            loss = loss.mean()

        # Log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        if labels.size(0) == bsz:
            accuracy = 100 * (predicted_idx == labels).float().mean()
        else:
            accuracy = 0
            start = 0
            for i, _len in enumerate(len_labels):
                pos_indices = labels[start:start+_len]
                if predicted_idx[i].item() in pos_indices:
                    accuracy += 1
                start += _len
            accuracy = 100 * (float(accuracy) / len_labels.size(0))
            
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats


class InBatchINFLinearProjection(nn.Module):
    """Linear Projection mode for INF-Retriever"""
    
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(InBatchINFLinearProjection, self).__init__()
        
        self.opt = opt
        self.norm_doc = opt.norm_doc if hasattr(opt, 'norm_doc') else True
        self.norm_query = opt.norm_query if hasattr(opt, 'norm_query') else True
        self.label_smoothing = opt.label_smoothing if hasattr(opt, 'label_smoothing') else 0.0
        
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id,
                pooling=opt.pooling if hasattr(opt, 'pooling') else "last_token",
                random_init=opt.random_init if hasattr(opt, 'random_init') else False
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
        
        # INF-Retriever uses 1536 dimensions
        self.hidden_size = 1536
        self.linear = nn.Linear(2 * self.hidden_size, self.hidden_size)
    
    def _load_retriever(self, model_id, pooling, random_init):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        
        base_model = transformers.AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        
        retriever = inf_retriever.INFRetriever(base_model, pooling=pooling)
        
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
                    q_doc_embs.append(qemb[i].unsqueeze(0))

            newqemb = torch.cat(q_doc_embs, dim=0)
            assert newqemb.size() == qemb.size()
            qemb = newqemb
        
        gather_fn = dist_utils.gather
        gather_kemb = gather_fn(kemb)
        labels = labels + dist_utils.get_rank() * len(kemb)
        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)
        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        # Log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)