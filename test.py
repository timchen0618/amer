from logging import config
from re import A
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
    ContrastiveTrainCollator,
    DataHandler,
    contrastive_eval_collator
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

from gen_ret_and_eval import evaluate_loop
from data_creation.gaussian.baseline_evaluation import evaluate_baseline

def load_synthetic_dataset(data_dir='./synthetic_data', split='test'):
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
    
    pairs_data = pairs_data[split]
    # data = load_synthetic_dataset(data_dir='data_creation/gaussian/data/opposing_pairs_data/')
    
    print(len(pairs_data)) # dict_keys(['query_idx', 'query_vector', 'ground_truth_indices'])
    print(pairs_data[0]['query_idx']) 
    print(np.array(pairs_data[0]['query_vector']).shape)
    print(np.array(pairs_data[0]['ground_truth_indices']).shape)
    return pairs_data, queries, corpus
    
    # return {
    #     'config': config,
    #     'corpus': corpus,
    #     'queries': queries,
    #     'transformation_matrices': transformation_matrices,
    #     'pairs_data': pairs_data
    # }


def load_model_local(adapter_path=None, linear_checkpoint_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model(train_lora=True,
                                    base_model_id="meta-llama/Llama-3.2-1B-Instruct", 
                                    adapter_path=adapter_path, 
                                    linear_checkpoint_path=linear_checkpoint_path,
                                    embedding_model_dim=1024, 
                                    weight_tying=False, 
                                    loss_function='Hungarian_Contrastive', 
                                    temperature=0.05,
                                    extra_q_embed=False,
                                    compute_loss_on_q=False,
                                    use_eos=False)
    model.to(device)
    return model, tokenizer, device

def load_input_data(input_data_path):
    # Load dataset
    collator = contrastive_eval_collator
    full_dataset = load_embeddings_dataset(dataset_path=input_data_path)
    data_handler = DataHandler(full_dataset, collator, 1, 'dev')
    
    dataloader = data_handler.get_full_dataloader()
    return dataloader

    
command = 'simple_eval'




if command == 'simple_train':
    test_pairs, queries, corpus = load_synthetic_dataset(data_dir='data_creation/gaussian/data/opposing_pairs_data/')
    model, tokenizer, device = load_model_local()
    

    batch = {'inputs_embeds': [], 'attention_mask':[], 'positive_embeddings': [], 'negative_embeddings': []}

    LENGTH = 16
    for i in range(len(test_pairs)):
        query_vector = queries[test_pairs[i]['query_idx']]
        ground_truth_indices = test_pairs[i]['ground_truth_indices']
        ground_truth_embeddings = corpus[ground_truth_indices]
        batch['inputs_embeds'].append(query_vector)
        batch['attention_mask'].append(np.zeros(LENGTH))
        batch['positive_embeddings'].append(ground_truth_embeddings)
        batch['negative_embeddings'].append(ground_truth_embeddings)

    batch['inputs_embeds'] = torch.tensor(batch['inputs_embeds']).to(device).float().unsqueeze(1).expand(-1, LENGTH, -1)
    batch['attention_mask'] = torch.tensor(batch['attention_mask']).to(device).long()
    batch['attention_mask'][:, 0] = 1
    batch['positive_embeddings'] = torch.tensor(batch['positive_embeddings']).to(device).float()
    batch['negative_embeddings'] = torch.tensor(batch['negative_embeddings']).to(device).float()
    print('input embeds', batch['inputs_embeds'].size(), 'attention mask', batch['attention_mask'].size(), 'positive embeddings', batch['positive_embeddings'].size(), 'negative embeddings', batch['negative_embeddings'].size())
    
    outputs = model(**batch)
    print('loss', outputs.loss)
    
    
elif command == 'simple_eval':
    split = 'train'
    data_dir = 'data_creation/gaussian/data/opposing_pairs_data/'
    
    pairs_data = json.load(open(os.path.join(data_dir, 'query_ground_truth_pairs.json'), 'r'))
    test_pairs, queries, corpus = load_synthetic_dataset(data_dir=data_dir, split=split)
    # evaluate_loop(dataloader, model, device, max_new_tokens, use_gt_q_embed, use_eos, compute_loss = True)
    
    adapter_path = 'results/gaussian_synthetic_inf/toy_contrastive_lr1e4_ep30_temp0.05_warmup0.05_gradnorm1_bs16/best_model.pt'
    linear_checkpoint_path = 'results/gaussian_synthetic_inf/toy_contrastive_lr1e4_ep30_temp0.05_warmup0.05_gradnorm1_bs16/best_model_linear.pt'
    model, tokenizer, device = load_model_local(adapter_path=adapter_path, linear_checkpoint_path=linear_checkpoint_path)
    
    with torch.no_grad():
        dataloader = load_input_data(f'training_datasets/gaussian_synthetic/inf/gaussian_synthetic_{split}_dataset_1b_contrastive/')
        # batch = next(iter(dataloader))
        # outputs = model.generate(**batch, max_new_tokens=1)
        # print(outputs.shape)
        all_outputs, _, _, all_lengths = evaluate_loop(dataloader, model, device, max_new_tokens=1, use_gt_q_embed=False, use_eos=False, compute_loss=False)
        print(all_outputs.shape)
        print(all_lengths)

    # avg_results = evaluate_baseline("Average Baseline", avg_predictions, corpus, pairs_data, args.k_values)
    results = evaluate_baseline("Trained Model", all_outputs, corpus, pairs_data, [10, 20, 50, 100, 200, 500])






