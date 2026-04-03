# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# contriever.py
import os
import torch
import transformers

from src import utils, inbatch


# Add this to your contriever.py file

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class QwenRetriever(nn.Module):
    def __init__(self, config, pooling="last_token", **kwargs):  # Changed default pooling
        super().__init__()
        self.config = config
        self.config.pooling = pooling
        
        # Load the base Qwen model
        self.qwen_model = AutoModel.from_pretrained(
            config.name_or_path, 
            trust_remote_code=True,
            # attn_implementation="flash_attention_2", 
            # torch_dtype=torch.float16
            torch_dtype=torch.bfloat16,
            use_cache=False
        )
        
        # Get hidden size from config
        self.hidden_size = 1024
        self.qwen_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=True,  # Changed default to True
    ):
        # Forward pass through Qwen model - only pass supported args
        outputs = self.qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # Get last hidden states
        last_hidden = outputs.last_hidden_state
        
        #Apply pooling strategy
        # if self.config.pooling == "last_token" or self.config.pooling == "last":
        #     # Last token pooling (standard for Qwen embeddings)
        #     #print("last token pooling", flush=True)
        #     left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        #     if left_padding:
        #         emb = last_hidden[:, -1]
        #     else:
        #         batch_size = last_hidden.shape[0]
        #         sequence_lengths = attention_mask.sum(dim=1) - 1
        #         emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
        # elif self.config.pooling == "average":
        #     # Average pooling
        #     last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        #     emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        # elif self.config.pooling == "cls":
        #     # First token
        #     emb = last_hidden[:, 0]
        # else:
        #     raise ValueError(f"Unknown pooling strategy: {self.config.pooling}")
        
        # NOTE: right pooling
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

        emb = emb.contiguous()
        if normalize:
            emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
            
        return emb
    
def load_retriever(model_path, pooling="last_token", random_init=False):
    # Check if model exists locally
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu",weights_only=False)
        opt = pretrained_dict["opt"]
        if hasattr(opt, "retriever_model_id"):
            retriever_model_id = opt.retriever_model_id
        else:
            # Default to Qwen3 embedding model
            retriever_model_id = "Qwen/Qwen3-Embedding-0.6B"
            
        tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_model_id, trust_remote_code=True)
        cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        cfg.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "right"
        
        
        # Add model path to config for QwenRetriever
        cfg.name_or_path = retriever_model_id
        cfg.pooling = pooling

        if hasattr(opt, "run_name"):
            if hasattr(opt, "training_mode"):
                print(f"Training mode: {opt.training_mode}")
                if opt.training_mode in ['standard', 'base']:
                    model_class = inbatch.InBatch
                elif opt.training_mode == 'gru':
                    model_class = inbatch.InBatchGRU
                elif opt.training_mode == 'linear_projection':
                    model_class = inbatch.InBatchLinearProjection
                else:
                    model_class = inbatch.InBatch
            else:
                raise NotImplementedError("training_mode not specified")
        else:
            model_class = QwenRetriever
            
        print(f"Using model class: {model_class}, retriever_model_id: {retriever_model_id}")
        pretrained_dict = pretrained_dict["model"]
        
        if model_class == QwenRetriever:
            retriever = model_class(cfg, pooling=pooling)
            retriever.load_state_dict(pretrained_dict, strict=True)
        else:
            # Load InBatch models
            if opt.training_mode == 'standard':
                model = inbatch.InBatch(opt, None, None)
                print("Using model = InBatch", flush=True)
            elif opt.training_mode == 'base':
                model = inbatch.InBatch(opt, None, None)
                print("Using model = InBatch for base mode", flush=True)
            elif opt.training_mode == 'gru':
                model = inbatch.InBatchGRU(opt, None, None)
                print("Using model = InBatchGRU", flush=True)
            elif opt.training_mode == 'linear_projection':
                model = inbatch.InBatchLinearProjection(opt, None, None)
                print("Using model = InBatchLinearProjection", flush=True)
            else:
                model = inbatch.InBatch(opt, None, None)

            model.load_state_dict(pretrained_dict, strict=True)
            model.eval()
            print('Finished loading model')
            
            if opt.training_mode in ['standard']:
                retriever = model.encoder
            else:
                retriever = model
        
    else:
        # Loading from HuggingFace
        retriever_model_id = model_path
        print(f"Loading model from HuggingFace: {retriever_model_id}")
        
        # Load config and tokenizer with trust_remote_code=True for Qwen
        cfg = utils.load_hf(transformers.AutoConfig, model_path, trust_remote_code=True)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        cfg.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = 'right'
        
        # Add model path to config
        cfg.name_or_path = model_path
        cfg.pooling = pooling
        
        # Initialize QwenRetriever
        retriever = QwenRetriever(cfg, pooling=pooling)

    return retriever, tokenizer, retriever_model_id