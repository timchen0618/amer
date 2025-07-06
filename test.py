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
import numpy as np

def load_synthetic_dataset(data_dir='./synthetic_data'):
    """
    Load the complete synthetic dataset.
    
    Args:
        data_dir: Path to the directory containing the synthetic data files
        
    Returns:
        Dictionary containing all loaded components
    """
    
    # 1. Load configuration (metadata about the dataset)
    with open(os.path.join(data_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # 2. Load the main data arrays
    corpus = np.load(os.path.join(data_dir, 'corpus.npy'))              # Shape: (corpus_size, dimensions)
    queries = np.load(os.path.join(data_dir, 'queries.npy'))            # Shape: (total_queries, dimensions)
    transformation_matrices = np.load(os.path.join(data_dir, 'transformation_matrices.npy'))  # Shape: (n_rotations, dimensions, dimensions)
    
    # 3. Load query-ground truth mappings
    with open(os.path.join(data_dir, 'query_ground_truth_pairs.json'), 'r') as f:
        pairs_data = json.load(f)
    
    return {
        'config': config,
        'corpus': corpus,
        'queries': queries,
        'transformation_matrices': transformation_matrices,
        'pairs_data': pairs_data
    }

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = load_synthetic_dataset(data_dir='data_creation/gaussian/data/opposing_pairs_data/')
test_pairs = data['pairs_data']['test']
print(len(test_pairs)) # dict_keys(['query_idx', 'query_vector', 'ground_truth_indices'])
print(test_pairs[0]['query_idx']) 
print(np.array(test_pairs[0]['query_vector']).shape)
print(np.array(test_pairs[0]['ground_truth_indices']).shape)
queries = data['queries']
print(queries[test_pairs[0]['query_idx']])
# model, tokenizer = load_model(train_lora=True,
#                                 base_model_id="meta-llama/Llama-3.2-1B-Instruct", 
#                                 adapter_path="", 
#                                 linear_checkpoint_path=None,
#                                 embedding_model_dim=1536, 
#                                 weight_tying=False, 
#                                 loss_function='Hungarian_Contrastive', 
#                                 temperature=0.05,
#                                 extra_q_embed=False,
#                                 compute_loss_on_q=False,
#                                 use_eos=False)


# collator = functools.partial(ContrastiveTrainCollator(), shuffle=False, take_first=False, use_eos=False)
# full_dataset = load_embeddings_dataset(dataset_path='autoregressive_qampari_inf_train_dataset_1b_contrastive_5_ctxs')
# data_handler = DataHandler(full_dataset, collator, 1, 'train')
    
# train_dataloader, valid_loss_dataloader = data_handler.get_train_dev_dataloader(random_train_loader=False)

# total_length = len(train_dataloader)
# total_train_steps = 0

# num_epochs = 2
# total_steps = total_length * num_epochs
# warmup_steps = total_length * num_epochs * 0.05
# optimizer = torch.optim.AdamW(
#     model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
# )

# for epoch in range(num_epochs):

#     pbar = tqdm(
#         colour="blue",
#         desc=f"Training Epoch: {epoch+1}",
#         total=total_length,
#         dynamic_ncols=True,
#     )
    
#     # shuffle the data by length
#     data_handler.length_aware_shuffle()

#     train_dataloader = data_handler.get_sequential_train_dataloader()
    
#     for step, batch in enumerate(train_dataloader):
#         for k, v in batch.items():
#             batch[k] = v.to(device)
#             if step == 0 and k == 'labels':
#                 print('labels, 0', batch[k].size())
#             if step == 0:
#                 logger.info(k, size=batch[k].size())
#         total_train_steps += 1
#         outputs = model(**batch)

#         loss = outputs.loss
#         loss.backward()
        
#         # clip gradients
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#         optimizer.step()
#         optimizer.zero_grad()
#         pbar.update(1)

#         pbar.set_description(
#             f"Training Epoch: {epoch+1}/{num_epochs}, batch {step}/{len(train_dataloader)} "
#             f"completed (loss): {round(float(loss.detach().float() * 1), 4)}"
#         )
        
#         if (total_train_steps-1) % 10 == 0:
#             ## enter evaluation mode
#             total_loss = 0
#             with torch.no_grad():
#                 model.eval()
#                 for step, batch in enumerate(tqdm(valid_loss_dataloader)):
#                     for k, v in batch.items():
#                         batch[k] = v.to(device)
#                     outputs = model(**batch)
#                     loss = outputs.loss
#                     total_loss += loss.item()                    
#             best_val_loss = total_loss / len(valid_loss_dataloader)
#             # save_model(model, save_dir, total_train_steps, best_val_loss)

#             gc.collect()
#             torch.cuda.empty_cache()
#     pbar.close()

            






