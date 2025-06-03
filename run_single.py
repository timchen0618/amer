# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from logging import config
import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import wandb

import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


from model import EmbeddingModel
from dataset import (
    MyCollator,
    load_embeddings_dataset
)

from tqdm import tqdm
from copy import copy
import os, sys
import yaml
import json
import gc
import argparse
import functools
from utils import Config, set_seed, set_optim

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
    model = EmbeddingModel(model, start_id, tokenizer.eos_token_id)
    
    return model, tokenizer


def main():

    parser = argparse.ArgumentParser(description="EmbeddingModel")
    parser.add_argument("--config_file", type=str, default='')
    args = parser.parse_args()    

    # load the configuration file
    with open(args.config_file) as f:
        config_dict = yaml.safe_load(f)

    print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained(configs.model_id)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(configs.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    start_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")

    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            # task_type="CAUSAL_LM",
            target_modules="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj".split(",")
        )
 
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = EmbeddingModel(model, start_id, tokenizer.eos_token_id)


    total_train_steps = 0
    if not configs.debug and not configs.only_eval:
        wandb_run = wandb.init(project=configs.project, name=configs.name)
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    # data loading
    collator = MyCollator()
    dataset_train = load_embeddings_dataset(dataset_path='autoregressive_wsd_dev_dataset')
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        batch_size=configs.batch_size_training,
        collate_fn=collator,
        sampler=RandomSampler(dataset_train),
    )

    dataset_loss_val = load_embeddings_dataset(dataset_path='autoregressive_wsd_dev_dataset')
    valid_loss_dataloader = torch.utils.data.DataLoader(
        dataset_loss_val,
        num_workers=1,
        shuffle=False,
        pin_memory=True,
        batch_size=configs.batch_size_training,
        collate_fn=collator,
        sampler=SequentialSampler(dataset_loss_val),
    )
    total_length = len(train_dataloader) // configs.gradient_accumulation_steps
    
    
    # optimize and scheduler    
    configs.total_steps = total_length * configs.num_epochs
    configs.warmup_steps = total_length * configs.num_epochs * configs.warmup_ratio
    optimizer, scheduler = set_optim(configs, model)
    
    # train_dataloader, valid_loss_dataloader, model, optimizer, scheduler = accelerator.prepare(
    #     train_dataloader, valid_loss_dataloader, model, optimizer, scheduler
    # )
    model = model.to(device)
    
    

    best_acc = 0
    best_val_loss = 10000
    for epoch in range(configs.resume, configs.num_epochs):

        if not configs.only_eval:
            ##############
            # Do Training
            ##############
            
            pbar = tqdm(
                colour="blue",
                desc=f"Training Epoch: {epoch+1}",
                total=total_length,
                dynamic_ncols=True,
            )
            
            for step, batch in enumerate(train_dataloader):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                total_train_steps += 1
                outputs = model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()


                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    pbar.update(1)

                if wandb_run:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": loss.detach().float()
                        * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)

                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss: {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
            pbar.close()

            if (
                not configs.save_only_improve
                and not configs.debug
                and not configs.only_eval
            ):
            
                model.base_causallm.save_pretrained(os.path.join(save_dir, f"checkpoint_{epoch + 1}"), safe_serialization=True)
                print("saving model.")

                gc.collect()
                torch.cuda.empty_cache()

            # val loss
            total_loss = 0
            with torch.no_grad():
                model.eval()
                for step, batch in enumerate(valid_loss_dataloader):
                    for k, v in batch.items():
                        batch[k] = v.to(device)
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()

                if wandb_run:
                    log_dict = {
                        "eval/loss": total_loss / len(valid_loss_dataloader),
                    }
                    wandb_run.log(log_dict)
                    print("eval loss", total_loss / len(valid_loss_dataloader))

     
        if configs.only_eval:
            break

        
        ##############
        # Model Saving
        ##############
        if (
            total_loss / len(valid_loss_dataloader) < best_val_loss
            and configs.save_only_improve
            and not configs.debug
            and not configs.only_eval
        ):
            model.base_causallm.save_pretrained(os.path.join(save_dir, f"checkpoint_{epoch + 1}"), safe_serialization=True)
            print("saving model.")

            # best_acc = cor / total
            best_val_loss = total_loss / len(valid_loss_dataloader)

            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
    
    # cuda 4,5,6,7 torchrun --nproc_per_node=4 run.py configs/train.yaml