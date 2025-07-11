from logging import config
import torch
import torch.distributed
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from dataclasses import dataclass

from src.model import EmbeddingModel, load_model
from src.dataset import (
    load_embeddings_dataset,
    MSETrainCollator,
    ContrastiveTrainCollator,
    DataHandler
)
from src.utils import Config, set_seed, set_optim
from src.model import save_model_single
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



    


def train(configs):
    if not configs.debug:
        log_with_wandb = True
    else:
        log_with_wandb = False

    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # tracking
    if log_with_wandb:        
        os.environ['WANDB_INIT_TIMEOUT'] = '600'
        os.environ['WANDB_DEBUG'] = 'true'
        ttt = configs.save_path.strip('/').split('/')[-1]
        logger.info('tags', wandb_tag=ttt)
        wandb_run = wandb.init(project=configs.project, name=configs.name, settings=wandb.Settings(init_timeout=120), tags=[ttt])
        wandb_run.config.update(configs, allow_val_change=True)
    else:
        wandb_run = None

    # data loading
    if configs.loss_function == 'MSE' or configs.loss_function == 'Hungarian_MSE':
        collator = functools.partial(MSETrainCollator(), shuffle=True, question_only=configs.question_only)
    else:
        collator = functools.partial(ContrastiveTrainCollator(), shuffle=configs.shuffle_sequence, take_first=configs.take_first, use_eos=configs.use_eos)
    full_dataset = load_embeddings_dataset(dataset_path=configs.train_path)
    data_handler = DataHandler(full_dataset, collator, configs.batch_size_training, 'train')
        
    if configs.train_on_all_data:
        train_dataloader = data_handler.get_full_dataloader()
        valid_loss_dataloader = data_handler.get_full_dataloader()
    else:
        train_dataloader, valid_loss_dataloader = data_handler.get_train_dev_dataloader(random_train_loader=False)
    
    total_length = len(train_dataloader) // configs.gradient_accumulation_steps
    
    # model loading
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
                                  compute_loss_on_q=configs.compute_loss_on_q,
                                  use_eos=configs.use_eos)
    model = model.to(device)
    
    # optimize and scheduler    
    configs.total_steps = total_length * configs.num_epochs
    configs.warmup_steps = total_length * configs.num_epochs * configs.warmup_ratio
    optimizer, scheduler = set_optim(configs, model)

    
    total_train_steps = 0
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
                    # if k == 'inputs_embeds':
                        # print('inputs_embeds, 0', batch[k][:,:1])
                    # print(k, batch[k])
                    if step == 0:
                        logger.info(k, size=batch[k].size())
                total_train_steps += 1
                outputs = model(**batch)

                loss = outputs.loss / configs.gradient_accumulation_steps
                loss.backward()
                
                # clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=configs.max_grad_norm)
                
                if log_with_wandb:
                    losses.append(loss.detach().float())
                    
                    
                if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(
                    train_dataloader
                ) - 1:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)

                if log_with_wandb and (step) % 100 == 0:
                    log_dict = {
                        "train/epoch": epoch + 1,
                        "train/step": epoch * len(train_dataloader) + step,
                        "train/loss": sum(losses)/ len(losses) 
                        * configs.gradient_accumulation_steps,
                        "train/lr": scheduler.get_last_lr()[0]
                    }
                    wandb_run.log(log_dict, step=total_train_steps)
                    losses = []
                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                    f"completed (loss): {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                )
                
                
                if (total_train_steps-1) % configs.save_every_n_steps == 0:
                    ## enter evaluation mode
                    if (
                        not configs.save_only_improve
                        and not configs.debug
                        and not configs.only_eval
                    ):  
                        ## if save every n steps, save the model
                        save_model_single(model, save_dir, total_train_steps-1, 0, logger, configs.save_best_model)

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

                        if log_with_wandb:
                            log_dict = {
                                "eval/loss": (total_loss / len(valid_loss_dataloader)),
                            }
                            wandb_run.log(log_dict, step=total_train_steps)
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
                        best_val_loss = total_loss / len(valid_loss_dataloader)
                        save_model_single(model, save_dir, total_train_steps, best_val_loss, logger, configs.save_best_model)

                        gc.collect()
                        torch.cuda.empty_cache()
            pbar.close()

            

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train autoregressive model on a single GPU")
    
    # Main configuration
    parser.add_argument("--project", type=str, default="diverse_retrieval", help="Project name")
    parser.add_argument("--save_path", type=str, default="results/ambiguous_qe_inf/", help="Directory to save results")
    parser.add_argument("--name", type=str, default="toy_contrastive_4_gpus_from_stage2_lr2e5_ep20_temp0.05_warmup0.05", help="Experiment name")
    parser.add_argument("--train_path", type=str, default="training_datasets/ambiguous_qe/inf/autoregressive_ambiguous_qe_inf_train_dataset_1b_contrastive_2_to_5_ctxs/", help="Path to training dataset")
    
    # Save and load configuration
    parser.add_argument("--save_every_n_steps", type=int, default=50, help="Save model every n steps")
    parser.add_argument("--save_best_model", action="store_true", default=False, help="Save best model")
    parser.add_argument("--embedding_model_dim", type=int, default=1536, help="Embedding model dimension")
    parser.add_argument("--adapter_path", type=str, default="results/nq_inf/toy_contrastive/checkpoint_70000", help="Path to adapter checkpoint")
    parser.add_argument("--linear_checkpoint_path", type=str, default="results/nq_inf/toy_contrastive/checkpoint_70000_linear.pt", help="Path to linear checkpoint")

    # Training configuration
    parser.add_argument("--batch_size_training", type=int, default=32, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--scheduler", type=str, default="linear", help="Learning rate scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    
    # Training options
    parser.add_argument("--shuffle_sequence", action="store_true", default=True, help="Shuffle sequence during training")
    parser.add_argument("--train_on_all_data", action="store_true", default=False, help="Train on all available data")
    parser.add_argument("--save_only_improve", action="store_true", default=True, help="Save only when validation improves")
    parser.add_argument("--take_first", action="store_true", default=False, help="Take first sequence")
    parser.add_argument("--only_eval", action="store_true", default=False, help="Only run evaluation, skip training")
    
    # Model architecture
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for contrastive loss")
    parser.add_argument("--loss_function", type=str, default="Hungarian_Contrastive", 
                       choices=["MSE", "Hungarian_MSE", "Contrastive", "Hungarian_Contrastive"],
                       help="Loss function to use")
    parser.add_argument("--extra_q_embed", action="store_true", default=False, help="Use extra question embedding")
    parser.add_argument("--compute_loss_on_q", action="store_true", default=False, help="Compute loss on questions")
    parser.add_argument("--use_eos", action="store_true", default=False, help="Use EOS token")
    
    # Data loading
    parser.add_argument("--question_only", action="store_true", default=False, help="Use questions only")
    
    # Debug and advanced options
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode")
    parser.add_argument("--epochs_per_stage", type=int, default=3, help="Epochs per stage")
    parser.add_argument("--max_latent_stage", type=int, default=3, help="Maximum latent stage")
    parser.add_argument("--pad_latent_to_max", action="store_true", default=True, help="Pad latent to maximum")
    parser.add_argument("--uniform_prob", type=float, default=0.0, help="Uniform probability")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--resume", type=int, default=0, help="Resume from epoch")
    parser.add_argument("--reset_optimizer", action="store_true", default=True, help="Reset optimizer")
    
    # Optimizer configuration
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta2")
    parser.add_argument("--optim", type=str, default="adamw", help="Optimizer type")
    parser.add_argument("--lr_min_ratio", type=float, default=0.0, help="Minimum learning rate ratio")
    parser.add_argument("--eps", type=float, default=1e-6, help="Optimizer epsilon")
    parser.add_argument("--weight_tying", action="store_true", default=False, help="Use weight tying")
    
    args = parser.parse_args()
    
    logger.info("Config:", args=args)
    train(args)
    
    