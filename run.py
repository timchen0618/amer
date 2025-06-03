# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random, torch, os
import numpy as np
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import json
import itertools
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from datasets import Dataset
from datasets import load_from_disk
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


MAX_N_LATENT = 16

class Config:
    # to access a dict with object.key
    def __init__(self, dictionary):
        self.__dict__ = dictionary


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    

############ OPTIM


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio) * step / float(max(1, self.warmup))

        return max(
            0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup) / float(max(1.0, self.total - self.warmup)),
        )


class CosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio=0.1, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(CosineScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup:
            return float(step) / self.warmup
        s = float(step - self.warmup) / (self.total - self.warmup)
        return self.ratio + (1.0 - self.ratio) * math.cos(0.5 * math.pi * s)


def set_optim(opt, model):
    if opt.optim == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay
        )
    else:
        raise NotImplementedError("optimizer class not implemented")

    scheduler_args = {
        "warmup": opt.warmup_steps,
        "total": opt.total_steps,
        "ratio": opt.lr_min_ratio,
    }
    if opt.scheduler == "linear":
        scheduler_class = WarmupLinearScheduler
    elif opt.scheduler == "cosine":
        scheduler_class = CosineScheduler
    else:
        raise ValueError
    scheduler = scheduler_class(optimizer, **scheduler_args)
    return optimizer, scheduler




class EmbeddingModel(nn.Module):

    def __init__(
        self,
        base_causallm,
        # latent_token_id,
        start_latent_id,
        # end_latent_id,
        eos_token_id,
        embedding_model_dim,
    ):

        super(EmbeddingModel, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        # self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        # self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()
            
        hidden_size = self.base_causallm.config.hidden_size
        self.embedding_model_dim = embedding_model_dim
        self.input_projection = nn.Linear(embedding_model_dim, hidden_size).half()
        self.output_projection = nn.Linear(hidden_size, embedding_model_dim).half()
        
    def forward(self, **inputs):
        has_label = 'labels' in inputs
        if has_label:
            labels = inputs.pop("labels")
            # loss_mask = inputs.pop("loss_mask")
        assert has_label, "only support training now"
        loss_mask = inputs['attention_mask'].detach().clone()
        
        for i in range(inputs['hidden_states'].size(0)):
            hidden_states = inputs['hidden_states'][i].unsqueeze(0)
            labels_embed = self.input_projection(labels[i].unsqueeze(0))
            
            # assign the labels to the hidden states as input
            input_start_for_output = inputs['attention_mask'][i].sum()
            print(hidden_states.size(), labels_embed.size())
            print(labels[i].size())
            output_len = labels[i].size(0)
            print(input_start_for_output, output_len)
            hidden_states[:,input_start_for_output:input_start_for_output+output_len,:] = labels_embed
            
            # ignore the first token, which is the question representation using embedding model
            loss_mask[i,:input_start_for_output] = 0
            # fill out the loss mask and attention mask
            loss_mask[i,input_start_for_output:(input_start_for_output+output_len-1)] = 1
            print(loss_mask[i])
            assert loss_mask[i].sum() == (output_len - 1), (loss_mask[i].sum(), output_len)
            inputs['attention_mask'][i,input_start_for_output:(input_start_for_output+output_len)] = 1
            print(inputs['attention_mask'][i])
            assert inputs['attention_mask'][i].sum().item() == (input_start_for_output + output_len)
        
        inputs['inputs_embeds'] = inputs['hidden_states']
        del inputs['hidden_states']
        
        outputs = self.base_causallm(inputs_embeds=inputs['inputs_embeds'], attention_mask=inputs['attention_mask'], output_hidden_states=True)
        # hidden_states = outputs.last_hidden_state
        hidden_states = outputs.hidden_states[-1]
        
        if has_label:
            loss_fct = torch.nn.MSELoss()
            flatten_hidden_states = hidden_states.view(-1, hidden_states.size(-1))
            flatten_labels = labels.view(-1, labels.size(-1))
            loss = loss_fct(flatten_hidden_states * loss_mask.view(-1, 1), flatten_labels * loss_mask.view(-1, 1))
            return Outputs(loss=loss, inputs_embeds=inputs['inputs_embeds'], last_hidden_states=hidden_states)
        else:
            return Outputs(loss=None, inputs_embeds=inputs['inputs_embeds'], last_hidden_states=hidden_states)
    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=MAX_N_LATENT,
        # output_embedding=False,
        # synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        # tokens = input_ids[0].detach().tolist()

        # # HC COMMENT: only process the outputs to get the first tokens
        
        # labels = input_ids.clone()  # placeholder. not used.
        # outputs = self.forward(
        #     input_ids,
        #     torch.ones_like(input_ids, device=input_ids.device),
        #     labels,
        #     torch.arange(
        #         0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
        #     ).reshape(1, -1),
        # )
        # inputs_embeds = outputs.inputs_embeds
        
        
        # HC Implementation
        next_embs = []
        outputs = self.base_causallm(input_ids=input_ids, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[-1][:, -1, :]  # (bsz, length, dim)
        next_emb = hidden_states.unsqueeze(1)
        next_embs.append(next_emb)
        
        inputs_embeds = outputs['hidden_states'][0]  # (bsz, length, dim)
        new_inputs_embeds = torch.cat((inputs_embeds, next_emb), dim=1)
        # get the first embedding, taking the last position from the last hidden state
        
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds, output_hidden_states=True)
            self.gen_forward_cnt += 1
            next_emb = outputs.hidden_states[-1][:, -1, :].unsqueeze(1)
            next_embs.append(next_emb)
            new_inputs_embeds = torch.cat((new_inputs_embeds, next_emb), dim=1)
        
        out_embs = torch.cat(next_embs, dim=1)
        return out_embs
    
    
def load_model(base_model_id, adapter_path):
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.gradient_checkpointing_enable()
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    start_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    
    # Load the model with LoRA adapter weights
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Wrap with your custom EmbeddingModel
    model = EmbeddingModel(model, start_id, tokenizer.eos_token_id, 1536)
    
    return model, tokenizer




@dataclass
class MyCollator:
    # tokenizer: PreTrainedTokenizerBase
    # latent_id: Optional[int] = None
    # label_pad_token_id: Optional[int] = -100
    def __call__(self, features, return_tensors=None):
        batch = {}
        for k in features[0].keys():
            batch[k] = torch.tensor([f[k] for f in features])

        return batch


def load_embeddings_dataset(dataset_path='autoregressive_dev_dataset'):
    dataset = load_from_disk(dataset_path)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataset


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "last_hidden_states"])
MAX_N_LATENT = 16

