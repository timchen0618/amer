import torch
import torch.distributed
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed

from src.dataset import (
    load_embeddings_dataset,
    MSETrainCollator,
    ContrastiveTrainCollator,
    DataHandler
)
from src.model import load_model, save_model_distributed
from src.utils import set_optim
from src.option import get_training_args

from tqdm import tqdm
import os

import gc
import argparse
import functools
import structlog


logger = structlog.get_logger()

def train(configs):
    if not configs.debug:
        log_with_wandb = True
        log_with_string = 'wandb'
    else:
        log_with_wandb = False
        log_with_string = None  
        
    # Initialize accelerator
    accelerator = Accelerator(gradient_accumulation_steps=configs.gradient_accumulation_steps, log_with=log_with_string, mixed_precision='fp16')
    
    if log_with_wandb:
        if accelerator.is_main_process:
            os.environ['WANDB_INIT_TIMEOUT'] = '600'
            os.environ['WANDB_DEBUG'] = 'true'
            ttt = configs.save_path.strip('/').split('/')[-1]
            logger.info('tags', wandb_tag=ttt)
            wandb_run = wandb.init(project=configs.project, name=configs.name, settings=wandb.Settings(init_timeout=120), tags=[ttt, 'distributed'])
            wandb_run.config.update(configs, allow_val_change=True)
            text_table = wandb.Table(columns=["step", "text"])
        
        accelerator.init_trackers(
            project_name=configs.project,   # wandb project name
        )        
    
    set_seed(configs.seed)
    save_dir = os.path.join(configs.save_path, configs.name)
    
    if accelerator.is_main_process:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # data loading
    if configs.loss_function == 'MSE' or configs.loss_function == 'Hungarian_MSE':
        collator = functools.partial(MSETrainCollator(), shuffle=configs.shuffle_sequence, first_label_only=configs.first_label_only, left_padding=configs.left_padding)
    else:
        collator = functools.partial(ContrastiveTrainCollator(), shuffle=configs.shuffle_sequence, take_first=configs.take_first, left_padding=configs.left_padding, use_eos=configs.use_eos)
    full_dataset = load_embeddings_dataset(dataset_path=configs.train_path)
    data_handler = DataHandler(full_dataset, collator, configs.batch_size_training, 'train', int(accelerator.num_processes) * 4)
        
    if configs.train_on_all_data:
        train_dataloader = data_handler.get_full_dataloader()
        valid_loss_dataloader = data_handler.get_full_dataloader()
    else:
        train_dataloader, valid_loss_dataloader = data_handler.get_train_dev_dataloader(random_train_loader=False)
    
    total_length = len(train_dataloader) // configs.gradient_accumulation_steps
    total_length = total_length // accelerator.num_processes
    
    assert configs.schedule_sampling == (configs.model_type in ['EmbeddingModelSS', 'EmbeddingModelSSVariable', 'EmbeddingModelSSVariableLeftPad']), 'Schedule sampling is only supported for EmbeddingModelSS'
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    model, tokenizer = load_model(train_lora=(not configs.full_finetuning),
                                base_model_id=configs.model_id, 
                                adapter_path=configs.adapter_path, 
                                linear_checkpoint_path=configs.linear_checkpoint_path,
                                embedding_model_dim=configs.embedding_model_dim, 
                                weight_tying=configs.weight_tying, 
                                loss_function=configs.loss_function, 
                                temperature=configs.temperature,
                                extra_q_embed=configs.extra_q_embed,
                                compute_loss_on_q=configs.compute_loss_on_q,
                                use_eos=configs.use_eos,
                                model_type=configs.model_type)
    model = model.to(accelerator.device)

    # optimize and scheduler    
    configs.total_steps = total_length * configs.num_epochs
    configs.warmup_steps = total_length * configs.num_epochs * configs.warmup_ratio
    optimizer, scheduler = set_optim(configs, model)

    # Prepare everything with accelerator
    model, optimizer, train_dataloader, valid_loss_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_loss_dataloader, scheduler
    )    
    

    total_train_steps = 0
    best_val_loss = 10000
    losses = []
    for epoch in range(configs.resume, configs.num_epochs):
        ##############
        # Do Training
        ##############
        
        pbar = tqdm(
            colour="blue",
            desc=f"Training Epoch: {epoch+1}",
            total=total_length,
            dynamic_ncols=True,
            disable=not accelerator.is_main_process
        )
        
        # shuffle the data by length
        data_handler.length_aware_shuffle()
        if configs.train_on_all_data:
            train_dataloader = data_handler.get_full_dataloader()
        else:
            train_dataloader = data_handler.get_sequential_train_dataloader()
        
        train_dataloader = accelerator.prepare(train_dataloader)
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                if configs.schedule_sampling:
                    batch['sampling_rate'] = total_train_steps / float(configs.total_steps)
                
                total_train_steps += 1
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                # clip the gradient
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), configs.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if log_with_wandb:
                    losses.append(loss.detach().float().cpu().item())
                    if (step) % 100 == 0:
                        log_dict = {
                            "train/epoch": epoch + 1,
                            "train/step": epoch * len(train_dataloader) + step,
                            "train/loss": sum(losses)/ len(losses) 
                            * configs.gradient_accumulation_steps,
                        }
                        accelerator.log(log_dict, step=total_train_steps)
                        losses = []
                
                # update the progress bar
                if accelerator.is_main_process:
                    if (step + 1) % configs.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        pbar.update(1)
                    pbar.set_description(
                        f"Training Epoch: {epoch+1}/{configs.num_epochs}, batch {step}/{len(train_dataloader)} "
                        f"completed (loss): {round(float(loss.detach().float().cpu().item() * configs.gradient_accumulation_steps), 4)}"
                    )
                
                ####### Evaluation #######
                if (total_train_steps-1) % configs.save_every_n_steps == 0:
                    ## enter evaluation mode
                    if (
                        not configs.save_only_improve
                        and not configs.debug
                    ):
                        save_model_distributed(model, save_dir, total_train_steps-1, best_val_loss, accelerator, logger, configs.save_best_model)
                        gc.collect()
                        torch.cuda.empty_cache()

                    # val loss
                    total_loss_list = []
                    with torch.no_grad():
                        model.eval()
                        for step, batch in enumerate(tqdm(valid_loss_dataloader, disable=not accelerator.is_main_process)):
                            if configs.schedule_sampling:
                                batch['sampling_rate'] = total_train_steps / float(configs.total_steps)
                            outputs = model(**batch)
                            loss = outputs.loss
                            total_loss_list.append(loss.view(1,))

                        total_loss_across_gpus = torch.cat(total_loss_list, dim=0).sum()
                        total_losses = accelerator.gather(total_loss_across_gpus.view(1,))
                        
                        total_loss = total_losses.sum().cpu().item()
                        if log_with_wandb:
                            log_dict = {
                                "eval/loss": (total_loss / len(valid_loss_dataloader)),
                            }
                            accelerator.log(log_dict, step=total_train_steps)
                            if accelerator.is_main_process:
                                logger.info("eval loss", eval_loss=(total_loss / len(valid_loss_dataloader)), length=len(valid_loss_dataloader))
                        del total_loss_across_gpus
                        del total_loss_list
                        del total_losses
                        gc.collect()
                        torch.cuda.empty_cache()
                        
                    ##############
                    # Model Saving
                    ##############
                    if (
                        total_loss / len(valid_loss_dataloader) < best_val_loss
                        and configs.save_only_improve
                        and not configs.debug
                    ):
                        best_val_loss = total_loss / len(valid_loss_dataloader)
                        save_model_distributed(model, save_dir, total_train_steps, best_val_loss, accelerator, logger, configs.save_best_model)
                        

                    gc.collect()
                    torch.cuda.empty_cache()
                
        pbar.close()
            
    accelerator.end_training()

if __name__ == "__main__":
    parser = get_training_args()
    parser.description = "Train autoregressive model with distributed training"
    args = parser.parse_args()
    
    logger.info("Config:", args=args)
    train(args) 