from logging import config
import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from dataclasses import dataclass

from model import EmbeddingModel, load_model
from dataset import (
    load_embeddings_dataset,
    MSETrainCollator,
    ContrastiveTrainCollator,
    DataHandler
)
from utils import Config, set_seed, set_optim

from tqdm import tqdm
from copy import copy
import os, sys

import yaml
import json
import gc
import argparse
import functools
import random
import structlog
logger = structlog.get_logger()


def save_model(model, save_dir, step):
    # Save the base causal language model
    model.base_causallm.save_pretrained(os.path.join(save_dir, f"checkpoint_{step}"), safe_serialization=True)
    
    # Save the linear layers
    linear_layers = {
        'input_projection': model.input_projection.state_dict(),
        'output_projection': model.output_projection.state_dict()
    }
    torch.save(linear_layers, os.path.join(save_dir, f"checkpoint_{step}_linear.pt"))
    
    logger.info(f"saving model.", step=(step))
    


def train():

    # load the configuration file
    with open('configs/train.yaml') as f:
        config_dict = yaml.safe_load(f)

    logger.info("Config:", config_dict=config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model, tokenizer = load_model(train_lora=True,
                                  base_model_id=configs.model_id, 
                                  adapter_path=configs.adapter_path, 
                                  linear_checkpoint_path=configs.linear_checkpoint_path,
                                  embedding_model_dim=configs.embedding_model_dim, 
                                  weight_tying=configs.weight_tying, 
                                  loss_function=configs.loss_function, 
                                  temperature=configs.temperature,
                                  extra_q_embed=configs.extra_q_embed,
                                  compute_loss_on_q=configs.compute_loss_on_q)

    total_train_steps = 0
    if not configs.debug and not configs.only_eval:        
        
        os.environ['WANDB_INIT_TIMEOUT'] = '600'
        os.environ['WANDB_DEBUG'] = 'true'
        ttt = configs.save_path.strip('/').split('/')[-1]
        logger.info('tags', wandb_tag=ttt)
        wandb_run = wandb.init(project=configs.project, name=configs.name, settings=wandb.Settings(init_timeout=120), tags=[ttt])
        wandb_run.config.update(configs, allow_val_change=True)
        text_table = wandb.Table(columns=["step", "text"])

    else:
        wandb_run = None

    # data loading
    if configs.loss_function == 'MSE' or configs.loss_function == 'Hungarian_MSE':
        collator = functools.partial(MSETrainCollator(), shuffle=True, question_only=configs.question_only)
    else:
        collator = functools.partial(ContrastiveTrainCollator(), shuffle=configs.shuffle_sequence, take_first=configs.take_first)
    full_dataset = load_embeddings_dataset(dataset_path=configs.train_path)
    data_handler = DataHandler(full_dataset, collator, configs.batch_size_training, 'train')
        
    if configs.train_on_all_data:
        train_dataloader = data_handler.get_full_dataloader()
        valid_loss_dataloader = data_handler.get_full_dataloader()
    else:
        train_dataloader, valid_loss_dataloader = data_handler.get_train_dev_dataloader(random_train_loader=False)
    
    total_length = len(train_dataloader) // configs.gradient_accumulation_steps
    
    # optimize and scheduler    
    configs.total_steps = total_length * configs.num_epochs
    configs.warmup_steps = total_length * configs.num_epochs * configs.warmup_ratio
    optimizer, scheduler = set_optim(configs, model)
    
    model = model.to(device)
    
    # save_model(model, save_dir, total_train_steps)

    best_acc = 0
    best_val_loss = 10000
    losses = []
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
            
            # shuffle the data by length
            data_handler.length_aware_shuffle()
            if configs.train_on_all_data:
                train_dataloader = data_handler.get_full_dataloader()
            else:
                train_dataloader = data_handler.get_sequential_train_dataloader()
            
            for step, batch in enumerate(train_dataloader):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                    if step == 0 and k == 'labels':
                        print('labels, 0', batch[k].size())
                    if step == 0:
                        logger.info(k, size=batch[k].size())
                total_train_steps += 1
                outputs = model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()
                if wandb_run:
                    losses.append(loss.detach().float())
                    
                    
                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if wandb_run and (step) % 100 == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss (*1000)": 1000*sum(losses)/ len(losses) 
                        * configs.gradient_accumulation_steps,
                    }
                    wandb_run.log(log_dict)
                    losses = []
                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss (*1000): {round(float(loss.detach().float() * configs.gradient_accumulation_steps * 1000), 4)}"
                )
                
                
                if (total_train_steps-1) % configs.save_every_n_steps == 0:
                    ## enter evaluation mode
                    if (
                        not configs.save_only_improve
                        and not configs.debug
                        and not configs.only_eval
                    ):
                    
                        save_model(model, save_dir, total_train_steps-1)

                        gc.collect()
                        torch.cuda.empty_cache()

                    # val loss
                    total_loss = 0
                    with torch.no_grad():
                        model.eval()
                        for step, batch in enumerate(tqdm(valid_loss_dataloader)):
                            for k, v in batch.items():
                                batch[k] = v.to(device)
                            outputs = model(**batch)
                            loss = outputs.loss
                            total_loss += loss.item()

                        if wandb_run:
                            log_dict = {
                                "eval/loss (*1000)": (total_loss / len(valid_loss_dataloader)) * 1000,
                            }
                            wandb_run.log(log_dict)
                            logger.info("eval loss", eval_loss=(total_loss / len(valid_loss_dataloader)))
                            
                            
                            
                    ##############
                    # Model Saving
                    ##############
                    if (
                        total_loss / len(valid_loss_dataloader) < best_val_loss
                        and configs.save_only_improve
                        and not configs.debug
                        and not configs.only_eval
                    ):
                        save_model(model, save_dir, total_train_steps)

                        # best_acc = cor / total
                        best_val_loss = total_loss / len(valid_loss_dataloader)

                        gc.collect()
                        torch.cuda.empty_cache()
            pbar.close()

            

        

if __name__ == "__main__":
    train()
    
    