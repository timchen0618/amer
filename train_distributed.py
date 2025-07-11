import torch
import torch.distributed
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed

from dataset import (
    load_embeddings_dataset,
    MSETrainCollator,
    ContrastiveTrainCollator,
    DataHandler
)
from utils import set_optim

from tqdm import tqdm
import os

import gc
import argparse
import functools
import structlog
from src.model import load_model, save_model_distributed

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
            wandb_run = wandb.init(project=configs.project, name=configs.name, settings=wandb.Settings(init_timeout=120), tags=[ttt])
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
        collator = functools.partial(MSETrainCollator(), shuffle=True, first_label_only=configs.first_label_only)
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
    total_length = total_length // accelerator.num_processes
    
    
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
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
                    losses.append(loss.detach().float())
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
                        f"completed (loss): {round(float(loss.detach().float() * configs.gradient_accumulation_steps), 4)}"
                    )
                
                ####### Evaluation #######
                if (total_train_steps-1) % configs.save_every_n_steps == 0:
                    ## enter evaluation mode
                    if (
                        not configs.save_only_improve
                        and not configs.debug
                    ):
                        save_model_distributed(model, save_dir, total_train_steps-1, accelerator, logger, configs.save_best_model)
                        gc.collect()
                        torch.cuda.empty_cache()

                    # val loss
                    total_loss_list = []
                    with torch.no_grad():
                        model.eval()
                        for step, batch in enumerate(tqdm(valid_loss_dataloader, disable=not accelerator.is_main_process)):
                            outputs = model(**batch)
                            loss = outputs.loss
                            total_loss_list.append(loss.view(1,))

                        total_loss_across_gpus = torch.cat(total_loss_list, dim=0).sum()
                        total_losses = accelerator.gather(total_loss_across_gpus.view(1,))
                        total_loss = total_losses.sum().item()
                        if log_with_wandb:
                            log_dict = {
                                "eval/loss": (total_loss / len(valid_loss_dataloader)),
                            }
                            accelerator.log(log_dict, step=total_train_steps)
                            if accelerator.is_main_process:
                                logger.info("eval loss", eval_loss=(total_loss / len(valid_loss_dataloader)), length=len(valid_loss_dataloader))
                            
                    ##############
                    # Model Saving
                    ##############
                    if (
                        total_loss / len(valid_loss_dataloader) < best_val_loss
                        and configs.save_only_improve
                        and not configs.debug
                    ):
                        save_model_distributed(model, save_dir, total_train_steps, accelerator, logger, configs.save_best_model)
                        best_val_loss = total_loss / len(valid_loss_dataloader)

                    gc.collect()
                    torch.cuda.empty_cache()
                
        pbar.close()
            
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train autoregressive model with distributed training")
    
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
    # adapter_path: results/qampari_inf/toy_qemb_from_nq/checkpoint_30000
    # linear_checkpoint_path: results/qampari_inf/toy_qemb_from_nq/checkpoint_30000_linear.pt
    # adapter_path: 
    # linear_checkpoint_path: 

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
    
    # Model architecture
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for contrastive loss")
    parser.add_argument("--loss_function", type=str, default="Hungarian_Contrastive", 
                       choices=["MSE", "Hungarian_MSE", "Contrastive", "Hungarian_Contrastive"],
                       help="Loss function to use")
    parser.add_argument("--extra_q_embed", action="store_true", default=False, help="Use extra question embedding")
    parser.add_argument("--compute_loss_on_q", action="store_true", default=False, help="Compute loss on questions")
    parser.add_argument("--use_eos", action="store_true", default=False, help="Use EOS token")
    
    # Data loading
    parser.add_argument("--first_label_only", action="store_true", default=False, help="Use first label (question) only")
    
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
    
    # # Convert args to config dict for compatibility with existing code
    # config_dict = vars(args)
    # configs = Config(config_dict)
    logger.info("Config:", args=args)
    train(args) 