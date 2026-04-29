# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# inf_retriever.py
import os
import torch
import transformers

from src import utils, inbatch


# INF-Retriever model implementation

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class INFRetriever(nn.Module):
    def __init__(self, config, pooling="last_token", **kwargs):
        super().__init__()
        self.config = config
        self.config.pooling = pooling
        
        # Load the base INF model
        self.inf_model = AutoModel.from_pretrained(
            config.name_or_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            use_cache=False
        )
        
        # Get hidden size from config (1536 for INF-Retriever)
        self.hidden_size = 1536
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.inf_model, 'gradient_checkpointing_enable'):
            print("grad checkpoint enabled")
            self.inf_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            self.inf_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=True,
    ):
        # Forward pass through INF model
        outputs = self.inf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # Get last hidden states
        last_hidden = outputs.last_hidden_state
        if self.config.pooling in ("last_token", "last"):
            seq_lens = attention_mask.sum(dim=1) - 1
            emb = last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lens]
        elif self.config.pooling == "average":
            last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]
        else:
            raise ValueError(f"Unknown pooling: {self.config.pooling}")
        
        # Ensure contiguous memory layout
        emb = emb.contiguous()
        
        # Normalize if requested (default True for INF-Retriever)
        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            
        return emb
    
def load_retriever(model_path, pooling="last_token", random_init=False):
    # Check if model exists locally
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict["opt"]
        if hasattr(opt, "retriever_model_id"):
            retriever_model_id = opt.retriever_model_id
        else:
            # Default to INF-Retriever model
            retriever_model_id = "infly/inf-retriever-v1-1.5b"
            
        tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_model_id, trust_remote_code=True)
        cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        cfg.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "right"
        
        # Add model path to config for INFRetriever
        cfg.name_or_path = retriever_model_id
        cfg.pooling = pooling

        if hasattr(opt, "run_name"):
            if hasattr(opt, "training_mode"):
                print(f"Training mode: {opt.training_mode}")
                if opt.training_mode == 'standard_org_q':
                    model_class = inbatch.EmbeddingModelDocEncNoProjSingleQuery
                elif opt.training_mode == 'multi':
                    model_class = inbatch.EmbeddingModelDocEncNoProj
            else:
                raise NotImplementedError("training_mode not specified")
        else:
            model_class = INFRetriever
            
        print(f"Using model class: {model_class}, retriever_model_id: {retriever_model_id}")
        pretrained_dict = pretrained_dict["model"]
        
        if model_class == INFRetriever:
            retriever = model_class(cfg, pooling=pooling)
            retriever.load_state_dict(pretrained_dict, strict=True)
        else:
            # Load InBatch models
            # if opt.training_mode == 'standard_org_q':
            #     model = inbatch.InBatch(opt, None, None)
            #     print("Using model = InBatch", flush=True)
            # else:
            #     model = inbatch.InBatch(opt, None, None)
            model = model_class(opt, None, None)
            print(f"Using model = {model_class}", flush=True)
            model.load_state_dict(pretrained_dict, strict=True)
            model.eval()
            print('Finished loading model')
            
            if opt.training_mode == 'standard_org_q':
                retriever = model.encoder
            else:
                retriever = model
        
    else:
        # Loading from HuggingFace
        retriever_model_id = model_path
        print(f"Loading model from HuggingFace: {retriever_model_id}")
        
        # Load config and tokenizer with trust_remote_code=True for INF
        cfg = utils.load_hf(transformers.AutoConfig, model_path, trust_remote_code=True)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        cfg.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "right"
        # Add model path to config
        cfg.name_or_path = model_path
        cfg.pooling = pooling
        
        # Initialize INFRetriever
        retriever = INFRetriever(cfg, pooling=pooling)

    return retriever, tokenizer, retriever_model_id