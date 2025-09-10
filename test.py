from logging import config
from operator import index
from re import A
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
    ContrastiveTrainCollator,
    DataHandler,
    contrastive_eval_collator
)
from src.utils import Config, set_seed, set_optim

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

from data_creation.gaussian.eval_utils import compute_similarities_and_rankings, compute_recall_at_k, compute_mrecall_at_k
from gen_ret_and_eval import evaluate_loop
from typing import List, Dict, Any


def load_synthetic_dataset(data_dir='./synthetic_data', split='test', normalize=False, indexed_corpus=None):
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
    if normalize: # normalized_corpus.npy
        assert indexed_corpus is None
        corpus = np.load(os.path.join(data_dir, 'normalized_corpus.npy'))              # Shape: (corpus_size, dimensions)
        queries = np.load(os.path.join(data_dir, 'normalized_queries.npy'))            # Shape: (total_queries, dimensions)
    else:
        if indexed_corpus is None:
            corpus = np.load(os.path.join(data_dir, 'corpus.npy'))              # Shape: (corpus_size, dimensions)
        else:
            corpus = np.load(os.path.join(data_dir, f'indexed_corpus_{indexed_corpus}.npy')) 
        queries = np.load(os.path.join(data_dir, 'queries.npy'))            # Shape: (total_queries, dimensions)
    
    # 3. Load query-ground truth mappings
    with open(os.path.join(data_dir, 'query_ground_truth_pairs.json'), 'r') as f:
        pairs_data = json.load(f)
    
    pairs_data = pairs_data[split]
    
    print('number of queries', len(pairs_data)) # dict_keys(['query_idx', 'query_vector', 'ground_truth_indices'])
    print('first query idx', pairs_data[0]['query_idx']) 
    print('first query vector', np.array(pairs_data[0]['query_vector']).shape)
    print('first ground truth indices', np.array(pairs_data[0]['ground_truth_indices']).shape)
    return pairs_data, queries, corpus


def load_model_local(base_model_id="meta-llama/Llama-3.2-1B-Instruct", adapter_path=None, linear_checkpoint_path=None, model_type="EmbeddingModel", embedding_model_dim=1024):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model(train_lora=False,
                                    base_model_id=base_model_id, 
                                    adapter_path=adapter_path, 
                                    linear_checkpoint_path=linear_checkpoint_path,
                                    embedding_model_dim=embedding_model_dim, 
                                    weight_tying=False, 
                                    loss_function='Hungarian_Contrastive', 
                                    temperature=0.05,
                                    extra_q_embed=False,
                                    compute_loss_on_q=False,
                                    use_eos=False,
                                    model_type=model_type)
    model.to(device)
    return model, tokenizer, device

def load_input_data(input_data_path, use_ground_truth_for_eval=False):
    # Load dataset
    if use_ground_truth_for_eval:
        collator = ContrastiveTrainCollator()
    else:
        collator = contrastive_eval_collator
    full_dataset = load_embeddings_dataset(dataset_path=input_data_path)
    
    if use_ground_truth_for_eval:
        data_handler = DataHandler(full_dataset, collator, 128, 'dev', 4)
    else:
        data_handler = DataHandler(full_dataset, collator, 1, 'dev', 4)
    
    dataloader = data_handler.get_full_dataloader()
    return dataloader


def aggregate_rankings(all_rankings, max_new_tokens, max_k):
    if max_new_tokens == 1:
        return all_rankings
    all_rankings = all_rankings[:, :max_k]
    
    # all_rankings is a list of numpy arrays. Size (max_new_tokens * num_samples, topk)
    new_outputs = []
    assert len(all_rankings) % max_new_tokens == 0
    print(len(all_rankings), len(all_rankings) // max_new_tokens, max_k) # 5000, 1000, 100
    # Take tokens from outputs in a round robin fashion
    for i in range(len(all_rankings) // max_new_tokens):
        all_outputs_chunk = all_rankings[i * max_new_tokens:(i + 1) * max_new_tokens,:max_k // max_new_tokens]
        new_output_inst = []
        for j in range(max_k // max_new_tokens):
            new_output_inst.append(all_outputs_chunk[:, j])
        new_outputs.append(np.concatenate(new_output_inst, axis=0))
    return np.array(new_outputs)


def evaluate_baseline_with_aggregation(baseline_name: str, predictions: np.ndarray, corpus: np.ndarray, 
                     test_pairs: List[Dict[str, Any]], k_values: List[int], max_new_tokens: int) -> Dict[str, float]:
    """
    Evaluate a baseline approach with multiple k values.
    
    Args:
        baseline_name: Name of the baseline approach
        predictions: Predicted vectors for test queries
        corpus: Full corpus to search in
        test_pairs: Query-ground truth mapping data
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary of metric results
    """
    print(f"\n=== Evaluating {baseline_name} ===")
    
    # Compute similarities and rankings
    similarities, rankings = compute_similarities_and_rankings(predictions, corpus, max(k_values))
    print(similarities.shape, rankings.shape)
    max_k = max(k_values)
    if max_new_tokens > 1:
        rankings = aggregate_rankings(rankings, max_new_tokens, max_k)
        print('aggregated rankings', rankings.shape)
    
    return rankings
    # results = {}
    # # Evaluate for each k
    # for k in k_values:
    #     recall = compute_recall_at_k(rankings, test_pairs, k)
    #     mrecall = compute_mrecall_at_k(rankings, test_pairs, k)
        
    #     results[f'recall@{k}'] = recall
    #     results[f'mrecall@{k}'] = mrecall
        
    #     print(f"  Recall@{k}: {recall:.4f}")
    #     print(f"  MRecall@{k}: {mrecall:.4f}")
    
    # return results, rankings


def eval_metrics(rankings, test_pairs, k_values, _print=True):
    results = {}
    for k in k_values:
        recall = compute_recall_at_k(rankings, test_pairs, k)
        mrecall = compute_mrecall_at_k(rankings, test_pairs, k)
        
        results[f'recall@{k}'] = recall
        results[f'mrecall@{k}'] = mrecall
        if _print:
            print(f"  Recall@{k}: {recall:.4f}")
            print(f"  MRecall@{k}: {mrecall:.4f}")
    return results

def eval_on_each_gt(rankings, test_pairs, k_values, _print=True):
    ### Evaluate on each GT
    all_results = []
    for i in range(5):
        new_test_pairs = []
        for t in test_pairs:
            new_test_pairs.append(t.copy())
            new_test_pairs[-1]['ground_truth_indices'] = [t['ground_truth_indices'][i]]
        result_per_gt = eval_metrics(rankings, new_test_pairs, k_values, _print=False)
        all_results.append(result_per_gt)
    scores = [['Metric', 'GT 1', 'GT 2', 'GT 3', 'GT 4', 'GT 5']]
    for k in k_values:
        scores.append([f'Recall@{k}'] + ['%2.2f' % all_results[i][f'recall@{k}'] for i in range(5)])
        
    if _print:
        import prettytable
        table = prettytable.PrettyTable()
        table.field_names = scores[0]
        for row in scores[1:]:
            table.add_row(row)
        print(table)
    
    return all_results, scores


def write_tsv(data: List[List[str]], file_path: str):
    import csv
    with open(file_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(data)

# if command == 'simple_train':
#     test_pairs, queries, corpus = load_synthetic_dataset(data_dir='./data_creation/gaussian/data/opposing_pairs_data_large/')
#     model, tokenizer, device = load_model_local()
    

#     batch = {'inputs_embeds': [], 'attention_mask':[], 'positive_embeddings': [], 'negative_embeddings': []}

#     LENGTH = 16
#     for i in range(len(test_pairs)):
#         query_vector = queries[test_pairs[i]['query_idx']]
#         ground_truth_indices = test_pairs[i]['ground_truth_indices']
#         ground_truth_embeddings = corpus[ground_truth_indices]
#         batch['inputs_embeds'].append(query_vector)
#         batch['attention_mask'].append(np.zeros(LENGTH))
#         batch['positive_embeddings'].append(ground_truth_embeddings)
#         batch['negative_embeddings'].append(ground_truth_embeddings)

#     batch['inputs_embeds'] = torch.tensor(batch['inputs_embeds']).to(device).float().unsqueeze(1).expand(-1, LENGTH, -1)
#     batch['attention_mask'] = torch.tensor(batch['attention_mask']).to(device).long()
#     batch['attention_mask'][:, 0] = 1
#     batch['positive_embeddings'] = torch.tensor(batch['positive_embeddings']).to(device).float()
#     batch['negative_embeddings'] = torch.tensor(batch['negative_embeddings']).to(device).float()
#     print('input embeds', batch['inputs_embeds'].size(), 'attention mask', batch['attention_mask'].size(), 'positive embeddings', batch['positive_embeddings'].size(), 'negative embeddings', batch['negative_embeddings'].size())
    
#     outputs = model(**batch)
#     print('loss', outputs.loss)
def random_baseline(pairs_data, corpus, topk):
    # set seed
    np.random.seed(42)
    rankings = []
    for _ in range(len(pairs_data)):
        rankings.append(np.random.randint(0, len(corpus), size=topk))
    return np.array(rankings)
    
def main(args):
    # load data
    pairs_data = json.load(open(os.path.join(args.data_dir, 'query_ground_truth_pairs.json'), 'r'))
    test_pairs, queries, corpus = load_synthetic_dataset(data_dir=args.data_dir, split=args.split, normalize=args.normalize, indexed_corpus=args.indexed_corpus)
    # evaluate_loop(dataloader, model, device, max_new_tokens, use_gt_q_embed, use_eos, compute_loss = True)
    pred_length_labels_str = '_pred_length' if args.pred_length else ''

    all_results = [['model_name', 'mrecall@100', 'recall@100', 'mrecall@10', 'recall@10']]
    all_scores = []
    if args.run_random_baseline:
        random_rankings = random_baseline(test_pairs, corpus, args.k_values[-1])
        print('random rankings', random_rankings.shape)
        model_path = args.model_paths[0]
        results = {}
        # Evaluate for each k
        for k in args.k_values:
            recall = compute_recall_at_k(random_rankings, test_pairs, k)
            mrecall = compute_mrecall_at_k(random_rankings, test_pairs, k)
            
            results[f'recall@{k}'] = recall
            results[f'mrecall@{k}'] = mrecall
            print(f"  Recall@{k}: {recall:.4f}")
            print(f"  MRecall@{k}: {mrecall:.4f}")
        np.save(os.path.join(model_path, f'random_baseline_rankings.npy'), random_rankings)
        all_results.append([model_path+f'_random_baseline', results['mrecall@100'], results['recall@100'], results['mrecall@10'], results['recall@10']])
    else:
        for model_path in args.model_paths:
            # load model
            if args.full_finetuning:
                print('doing full finetuning')
                base_model_id = os.path.join(model_path, args.checkpoint_name)
                adapter_path = None
            else:
                base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
                adapter_path = os.path.join(model_path, args.checkpoint_name)
            linear_checkpoint_path = os.path.join(model_path, f'{args.checkpoint_name}_linear.pt')
            model, _, device = load_model_local(base_model_id=base_model_id, 
                                                adapter_path=adapter_path, 
                                                linear_checkpoint_path=linear_checkpoint_path, 
                                                model_type=args.model_type,
                                                embedding_model_dim=args.embedding_model_dim)

            for max_new_tokens in args.max_new_tokens_list:
                with torch.no_grad():
                    # Create data loader
                    if args.data_dir == './data_creation/gaussian/data/linear_large/' or args.data_dir == 'data_creation/gaussian/data/linear_large/' or args.data_dir == 'data_creation/gaussian/data/linear_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_linear/inf/gaussian_linear_{args.split}_dataset_1b_contrastive/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/linear/' or args.data_dir == 'data_creation/gaussian/data/linear/':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_linear/inf/gaussian_linear_{args.split}_dataset_1b_contrastive_sm/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/linear_2048/' or args.data_dir == 'data_creation/gaussian/data/linear_2048/':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_linear/inf/gaussian_linear_{args.split}_dataset_1b_contrastive_sm_2048/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/diverse_mlps_multi_query/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_multi_query/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_multi_query':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_diverse_mlps_multi_query/inf/gaussian_diverse_mlps_multi_query_{args.split}_dataset_1b_contrastive{pred_length_labels_str}_sm/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/diverse_mlps_multi_query_large/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_multi_query_large/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_multi_query_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_diverse_mlps_multi_query/inf/gaussian_diverse_mlps_multi_query_{args.split}_dataset_1b_contrastive{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/diverse_mlps_data/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_data/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_data':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_diverse_mlps/inf/gaussian_diverse_mlps_{args.split}_dataset_1b_contrastive{pred_length_labels_str}_sm/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)                    
                    elif args.data_dir == './data_creation/gaussian/data/diverse_mlps_large/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_large/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_diverse_mlps/inf/gaussian_diverse_mlps_{args.split}_dataset_1b_contrastive{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)                    
                    elif args.data_dir == './data_creation/gaussian/data/diverse_mlps_sample_transformation/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_sample_transformation/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_sample_transformation':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_diverse_mlps_sample_transformation/inf/gaussian_diverse_mlps_sample_transformation_{args.split}_dataset_1b_contrastive{pred_length_labels_str}_sm/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/diverse_mlps_sample_transformation_large/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_sample_transformation_large/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_sample_transformation_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_diverse_mlps_sample_transformation/inf/gaussian_diverse_mlps_sample_transformation_{args.split}_dataset_1b_contrastive{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/diverse_mlps_ood_large/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_ood_large/' or args.data_dir == 'data_creation/gaussian/data/diverse_mlps_ood_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_diverse_mlps_ood/inf/gaussian_diverse_mlps_ood_{args.split}_dataset_1b_contrastive{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/linear_multi_query/' or args.data_dir == 'data_creation/gaussian/data/linear_multi_query/' or args.data_dir == 'data_creation/gaussian/data/linear_multi_query':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_linear_multi_query/inf/gaussian_linear_multi_query_{args.split}_dataset_1b_contrastive_sm{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/linear_multi_query_large/' or args.data_dir == 'data_creation/gaussian/data/linear_multi_query_large/' or args.data_dir == 'data_creation/gaussian/data/linear_multi_query_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_linear_multi_query/inf/gaussian_linear_multi_query_{args.split}_dataset_1b_contrastive{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/linear_sample_transformation/' or args.data_dir == 'data_creation/gaussian/data/linear_sample_transformation/' or args.data_dir == 'data_creation/gaussian/data/linear_sample_transformation':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_linear_sample_transformation/inf/gaussian_linear_sample_transformation_{args.split}_dataset_1b_contrastive_sm{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/linear_sample_transformation_large/' or args.data_dir == 'data_creation/gaussian/data/linear_sample_transformation_large/' or args.data_dir == 'data_creation/gaussian/data/linear_sample_transformation_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_linear_sample_transformation/inf/gaussian_linear_sample_transformation_{args.split}_dataset_1b_contrastive{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/linear_ood/' or args.data_dir == 'data_creation/gaussian/data/linear_ood/' or args.data_dir == 'data_creation/gaussian/data/linear_ood':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_linear_ood/inf/gaussian_linear_ood_{args.split}_dataset_1b_contrastive{pred_length_labels_str}_sm/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/linear_ood_large/' or args.data_dir == 'data_creation/gaussian/data/linear_ood_large/' or args.data_dir == 'data_creation/gaussian/data/linear_ood_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_linear_ood/inf/gaussian_linear_ood_{args.split}_dataset_1b_contrastive{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/new_mlps_data_large/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_data_large/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_data_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_new_mlps/inf/gaussian_new_mlps_{args.split}_dataset_1b_contrastive{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/new_mlps_harder_data_large/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_harder_data_large/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_harder_data_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_new_mlps_harder/inf/gaussian_new_mlps_harder_{args.split}_dataset_1b_contrastive{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/new_mlps_rotation_large/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_rotation_large/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_rotation_large':
                        if args.pred_length:
                            print('reading from data_creation/gaussian_new_mlps_rotation_5_test_dataset_1b_contrastive_pred_length/')
                            dataloader = load_input_data(f'data_creation/gaussian_new_mlps_rotation_5_test_dataset_1b_contrastive_pred_length/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)    
                        else:
                            dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_new_mlps_rotation/inf/gaussian_new_mlps_rotation_{args.split}_dataset_1b_contrastive/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/new_mlps_rotation_large_2/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_rotation_large_2/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_rotation_large_2':
                        dataloader = load_input_data(f'data_creation/gaussian_new_mlps_rotation_2_test_dataset_1b_contrastive_pred_length/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/new_mlps_normal_large/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_normal_large/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_normal_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_new_mlps_normal/inf/gaussian_new_mlps_normal_{args.split}_dataset_1b_contrastive{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    elif args.data_dir == './data_creation/gaussian/data/new_mlps_opposite_large/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_opposite_large/' or args.data_dir == 'data_creation/gaussian/data/new_mlps_opposite_large':
                        dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_new_mlps_opposite/inf/gaussian_new_mlps_opposite_{args.split}_dataset_1b_contrastive{pred_length_labels_str}/', use_ground_truth_for_eval=args.use_ground_truth_for_eval)
                    else:
                        raise ValueError(f'Invalid data directory: {args.data_dir}')


                    if args.use_ground_truth_for_eval:
                        all_losses = []
                        i = 0
                        all_outputs = []
                        for batch in tqdm(dataloader):
                            for k, v in batch.items():
                                if i == 0:
                                    print(k, v.shape)
                                batch[k] = v.to(device)

                            output = model(**batch)
                            all_losses.append(output.loss.item())
                            all_outputs.append(output.last_hidden_states[:, :max_new_tokens].view(-1, output.last_hidden_states.size(-1)))
                            i += 1
                        print('all losses', sum(all_losses) / len(all_losses))
                        all_outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
                        print('all outputs', all_outputs.shape)
                    else:
                        # # Evaluate model
                        # data_iter = iter(dataloader)

                        # skip first n batches
                        # for _ in range(1000):
                        #     next(data_iter, None)
                        all_outputs, _, _, all_lengths = evaluate_loop(dataloader, model, device, max_new_tokens=max_new_tokens, use_gt_q_embed=False, use_eos=False, compute_loss=False)
                        print('all outputs', all_outputs.shape)
                        all_outputs = all_outputs

                    # Evaluate Results
                    rankings = evaluate_baseline_with_aggregation(model_path+f'_max_new_tokens_{max_new_tokens}', all_outputs, corpus, pairs_data[args.split], args.k_values, max_new_tokens)
                    results = eval_metrics(rankings, pairs_data[args.split], args.k_values)
                    _, scores = eval_on_each_gt(rankings, test_pairs, args.k_values, _print=True)
                    
                    print('rankings', rankings.shape)
                    np.save(os.path.join(model_path, f'max_new_tokens_{max_new_tokens}_rankings.npy'), rankings)
                    all_results.append([model_path+f'_max_new_tokens_{max_new_tokens}', results['mrecall@100'], results['recall@100'], results['mrecall@10'], results['recall@10']])
                    
                    scores[0].append('MRecall')
                    for j, k in enumerate(args.k_values):
                        scores[j+1].append(results[f'mrecall@{k}'])

                    _id = model_path.strip('/').split('/')[-1]
                    all_scores.append([_id.split('lr')[0].split('finetuning')[1].strip('_') + f'_mnt_{max_new_tokens}'])
                    all_scores.extend(scores)

    write_tsv(all_results, 'results.tsv')
    import pandas as pd
    ### Record Results     
    score_results = pd.DataFrame(all_scores)
    score_results.to_csv('results/gaussian_synthetic_inf/recall_per_gt.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs='+', default=['results/gaussian_synthetic_inf/gaussian_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep30_warmup0.05/'])
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--data_dir", "-d", type=str, default='./data_creation/gaussian/data/opposing_pairs_data_large/')
    parser.add_argument("--k_values", type=int, nargs='+', default=[1, 5, 10, 20, 50, 100, 500]) # 10, 20, 50, 100, 200, 500
    parser.add_argument("--checkpoint_name", "-c", type=str, default='checkpoint_2001')
    parser.add_argument("--max_new_tokens_list", "-n", type=int, nargs='+', default=[5])
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--indexed_corpus", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="EmbeddingModel")
    parser.add_argument("--run_random_baseline", action='store_true')
    parser.add_argument("--full_finetuning", action='store_true')
    parser.add_argument("--embedding_model_dim", type=int, default=1024)
    parser.add_argument("--use_ground_truth_for_eval", action='store_true')
    parser.add_argument("--pred_length", action='store_true')
    args = parser.parse_args()
    main(args)





