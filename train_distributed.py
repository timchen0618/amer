import torch
import torch.distributed

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
        if configs.log_with == 'wandb':
            import wandb
        elif configs.log_with == 'trackio':
            import trackio as wandb
        else:
            raise ValueError(f"Invalid log_with: {configs.log_with}")
        
        log_with_wandb = True
        log_with_string = configs.log_with
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
            if configs.log_with == 'wandb':
                wandb_run = wandb.init(project=configs.project, name=configs.name, settings=wandb.Settings(init_timeout=600), tags=[ttt, 'distributed'])
                wandb_run.config.update(configs, allow_val_change=True)
            # elif configs.log_with == 'trackio':
            #     wandb_run = wandb.init(project=configs.project, name=configs.name)
            #     configs.tags = [ttt, 'distributed']
            #     config_dict = vars(configs)
            #     wandb_run.config.update(config_dict, allow_val_change=True)
        
        if configs.log_with == 'wandb':
            accelerator.init_trackers(
                    project_name=configs.project,   # wandb project name,
                )        
        elif configs.log_with == 'trackio':
            init_kwargs = {
                "trackio": {
                    'name': configs.name
                }
            }
            configs.tags = [configs.save_path.strip('/').split('/')[-1], 'distributed']
            config_dict = vars(configs)
            accelerator.init_trackers(
                project_name=configs.project,   # trackio project name,
                config=config_dict, init_kwargs=init_kwargs
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
    data_handler = DataHandler(full_dataset, collator, configs.batch_size_training, 'train', int(accelerator.num_processes) * 2)
        
    if configs.train_on_all_data:
        train_dataloader = data_handler.get_full_dataloader()
        valid_loss_dataloader = data_handler.get_full_dataloader()
    else:
        train_dataloader, valid_loss_dataloader = data_handler.get_train_dev_dataloader(random_train_loader=False)
    
    total_length = len(train_dataloader) // configs.gradient_accumulation_steps
    # total_length = total_length // accelerator.num_processes
    
    
    assert configs.schedule_sampling == (configs.model_type in ['EmbeddingModelSS', 'EmbeddingModelSSVariable', 'EmbeddingModelSSVariableLeftPad', 'EmbeddingModelSSAddQ', 'EmbeddingModelSSAvgQ', 'EmbeddingModelSSPredLength', 'EmbeddingModelSSVariableLeftPadPredLength']), 'Schedule sampling is only supported for EmbeddingModelSS'
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
                                model_type=configs.model_type,
                                normalize_embeddings=configs.normalize_embeddings)
    model = model.to(accelerator.device)

    # optimize and scheduler    
    configs.total_steps = total_length * configs.num_epochs
    configs.warmup_steps = total_length * configs.num_epochs * configs.warmup_ratio
    optimizer, scheduler = set_optim(configs, model) # for setting up the optimizer and scheduler, the total steps is the total number of batches

    # for distributed training, the total steps is the total steps for one process
    configs.total_steps = configs.total_steps // accelerator.num_processes
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, valid_loss_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_loss_dataloader, scheduler
    )    
    

    total_train_steps = 0
    resume_step = None
    starting_epoch = 0
    best_val_loss = 10000
    losses = []
    ntp_losses = []
    
    # We need to load the checkpoint back in before training here with `load_state`
    # The total number of epochs is adjusted based on where the state is being loaded from,
    # as we assume continuation of the same training script
    if configs.resume_from_checkpoint:
        # if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
        #     accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        #     accelerator.load_state(args.resume_from_checkpoint)
        #     path = os.path.basename(args.resume_from_checkpoint)
        # else:
        # Get the most recent checkpoint
        dirs = [f.name for f in os.scandir(save_dir) if (f.is_dir() and 'checkpoint' in f.name)]
        accelerator.print('dirs', dirs)
        
        if len(dirs) != 0:
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            accelerator.print(f"Resumed from checkpoint: {path}")
            accelerator.load_state(path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]
            accelerator.print('training_difference', training_difference)
            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)
            accelerator.print('starting_epoch', starting_epoch)
            accelerator.print('resume_step', resume_step)
            
    for epoch in range(starting_epoch, configs.num_epochs):
        if configs.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step only if we are not using a stateful dataloader
            if not configs.use_stateful_dataloader:
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            total_train_steps += resume_step
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = train_dataloader
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
                    # if configs.less_ss:
                    #     batch['sampling_rate'] = min((total_train_steps*5 / float(configs.total_steps)), 1.0)
                    # else:
                    batch['sampling_rate'] = min(configs.sample_rate_multiplier * total_train_steps / float(configs.total_steps), 0.8)
                    # print('sampling rate', batch['sampling_rate'], 'total_train_steps', total_train_steps, 'configs.total_steps', configs.total_steps)
                    
                if configs.force_sampling:
                    batch['sampling_rate'] = 1.0
                
                total_train_steps += 1
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                
                grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item()**2
                grad_norm = grad_norm**0.5

                
                # clip the gradient
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), configs.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if log_with_wandb:
                    losses.append(loss.detach().float().cpu().item())
                    if configs.pred_length:
                        ntp_losses.append(outputs.ntp_loss.detach().float().cpu().item())
                    if (step) % 100 == 0:
                        log_dict = {
                            "train/epoch": epoch + 1,
                            "train/step": epoch * len(train_dataloader) + step,
                            "train/loss": sum(losses)/ len(losses) 
                            * configs.gradient_accumulation_steps,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/grad_norm": grad_norm
                        }
                        if configs.schedule_sampling:
                            log_dict["train/sampling_rate"] = batch['sampling_rate']
                        if configs.pred_length:
                            log_dict["train/ntp_loss"] = sum(ntp_losses)/ len(ntp_losses)
                        accelerator.log(log_dict, step=total_train_steps)
                        losses = []
                        ntp_losses = []
                
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
                    # save model every n steps
                    # We save the model, optimizer, lr_scheduler, and seed states by calling `save_state`
                    # These are saved to folders named `step_{overall_step}`
                    # Will contain files: "pytorch_model.bin", "optimizer.bin", "scheduler.bin", and "random_states.pkl"
                    # If mixed precision was used, will also save a "scalar.bin" file
                    if configs.resume_from_checkpoint:
                        accelerator.print('saving checkpoint to ', os.path.join(save_dir, 'checkpoint'))
                        if isinstance(configs.save_every_n_steps, int):
                            accelerator.print('saving checkpoint to ', os.path.join(save_dir, 'checkpoint', f"step_{total_train_steps}"))
                            output_dir = f"step_{total_train_steps}"
                            if save_dir is not None:
                                output_dir = os.path.join(save_dir, output_dir)
                            accelerator.save_state(output_dir)
                    
                    
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
                    total_ntp_loss_list = []
                    with torch.no_grad():
                        model.eval()
                        for step, batch in enumerate(tqdm(valid_loss_dataloader, disable=not accelerator.is_main_process)):
                            if configs.schedule_sampling:
                                # batch['sampling_rate'] = total_train_steps / float(configs.total_steps)
                                batch['sampling_rate'] = 1.0
                            outputs = model(**batch)
                            loss = outputs.loss
                            total_loss_list.append(loss.view(1,))
                            if configs.pred_length:
                                total_ntp_loss_list.append(outputs.ntp_loss.view(1,))

                        total_loss_across_gpus = torch.cat(total_loss_list, dim=0).sum()
                        total_losses = accelerator.gather(total_loss_across_gpus.view(1,))
                        total_loss = total_losses.sum().cpu().item()
                        if configs.pred_length:
                            total_ntp_loss_across_gpus = torch.cat(total_ntp_loss_list, dim=0).sum()
                            total_ntp_losses = accelerator.gather(total_ntp_loss_across_gpus.view(1,))
                            total_ntp_loss = total_ntp_losses.sum().cpu().item()
                        if log_with_wandb:
                            log_dict = {
                                "eval/loss": (total_loss / len(valid_loss_dataloader)),
                            }
                            if configs.pred_length:
                                log_dict["eval/ntp_loss"] = (total_ntp_loss / len(valid_loss_dataloader))
                            accelerator.log(log_dict, step=total_train_steps)
                            if accelerator.is_main_process:
                                logger.info("eval loss", eval_loss=(total_loss / len(valid_loss_dataloader)), length=len(valid_loss_dataloader))
                                if configs.pred_length:
                                    logger.info("eval ntp_loss", eval_ntp_loss=(total_ntp_loss / len(valid_loss_dataloader)))
                        del total_loss_across_gpus
                        del total_loss_list
                        del total_losses
                        if configs.pred_length:
                            del total_ntp_loss_list
                            del total_ntp_losses
                            del total_ntp_loss
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
        
        # save model every epoch
        if configs.resume_from_checkpoint:
            if configs.save_every_n_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if save_dir is not None:
                    output_dir = os.path.join(save_dir, output_dir)
                accelerator.save_state(output_dir)
            
    accelerator.end_training()

if __name__ == "__main__":
    parser = get_training_args()
    parser.description = "Train autoregressive model with distributed training"
    args = parser.parse_args()
    
    logger.info("Config:", args=args)
    train(args) 