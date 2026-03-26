import torch
from src.model import load_model
from src.dataset import (
    load_embeddings_dataset,
    ContrastiveTrainCollator,
    DataHandler,
    contrastive_eval_collator
)

from tqdm import tqdm
import os

import json
import argparse
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
    max_k = max(k_values)
    if max_new_tokens > 1:
        rankings = aggregate_rankings(rankings, max_new_tokens, max_k)
    
    return rankings



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

def eval_on_each_gt(rankings, test_pairs, k_values, _print=True, num_gt=5):
    ### Evaluate on each GT
    all_results = []
    for i in range(num_gt):
        new_test_pairs = []
        for t in test_pairs:
            new_test_pairs.append(t.copy())
            new_test_pairs[-1]['ground_truth_indices'] = [t['ground_truth_indices'][i]]
        result_per_gt = eval_metrics(rankings, new_test_pairs, k_values, _print=False)
        all_results.append(result_per_gt)
    
    scores = [['Metric'] + [f'GT {i+1}' for i in range(num_gt)]]
    for k in k_values:
        scores.append([f'Recall@{k}'] + ['%2.2f' % all_results[i][f'recall@{k}'] for i in range(num_gt)])
        
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

def random_baseline(pairs_data, corpus, topk):
    # set seed
    np.random.seed(42)
    rankings = []
    for _ in range(len(pairs_data)):
        rankings.append(np.random.randint(0, len(corpus), size=topk))
    return np.array(rankings)
    
    
######## For Similarity Analysis ########
import random
import prettytable

""" 
    The script is used to check the distance between target vector embeddings.
    And also used to check the distance between predicted question embeddings. 
"""

def normalize_np(x, p=2, dim=1, eps=1e-12):
    """
    NumPy implementation of torch.nn.functional.normalize
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    return x / norm


def compute_l2_distance(query_1, query_2):
    return np.linalg.norm(query_1 - query_2)

def compute_cosine_similarity(query_1, query_2):
    return np.dot(query_1, query_2) / (np.linalg.norm(query_1) * np.linalg.norm(query_2))

    
def compute_averge_target_distance_same_example(target_vectors_list):
    l2_distance_list = []
    cosine_similarity_list = []
    for i in range(len(target_vectors_list)):
        all_target_vectors = target_vectors_list[i]
        for j in range(len(all_target_vectors)):
            for k in range(j+1, len(all_target_vectors)):
                l2_distance_list.append(compute_l2_distance(all_target_vectors[j], all_target_vectors[k]))
                cosine_similarity_list.append(compute_cosine_similarity(all_target_vectors[j], all_target_vectors[k]))
    return np.mean(l2_distance_list), np.mean(cosine_similarity_list)

def compute_averge_target_distance_different_examples(target_vectors_list, in_example_idx=0, _print=False):
    
    l2_distance_list = []
    cosine_similarity_list = []
    random_numbers = []
    for i in range(10000):
        two_random_nums = random.sample(range(len(target_vectors_list)), 2)
        random_numbers.append(two_random_nums)
        targets_1 = target_vectors_list[two_random_nums[0]]
        targets_2 = target_vectors_list[two_random_nums[1]]
        if in_example_idx != -1:
            query_1 = targets_1[in_example_idx]
            query_2 = targets_2[in_example_idx]
        else:
            if min(len(targets_1), len(targets_2)) == 0:
                continue
            else:
                query_1 = random.choice(targets_1)
                query_2 = random.choice(targets_2)
        l2_distance_list.append(compute_l2_distance(query_1, query_2))
        cosine_similarity_list.append(compute_cosine_similarity(query_1, query_2))
    if _print:
        print(random_numbers[:100])
        print(l2_distance_list[:100])
    return np.mean(l2_distance_list), np.mean(cosine_similarity_list)

def similarity_analysis(target_vectors_list):
    import sys
    from pathlib import Path
    
    all_l2_distances = []
    all_cosine_similarities = []
    table = prettytable.PrettyTable()
    table.field_names = ["distance_type", "betw ts (same)", "betw ts (diff), -1"]

    # compute the average distance between target vectors (of same example)
    l2_distance, cosine_similarity = compute_averge_target_distance_same_example(target_vectors_list)
    all_l2_distances.append(l2_distance.round(3))
    all_cosine_similarities.append(cosine_similarity.round(3))
    
    # compute the average distance between target vectors (of different examples)
    l2_distance, cosine_similarity = compute_averge_target_distance_different_examples(target_vectors_list, in_example_idx=-1)
    all_l2_distances.append(l2_distance.round(3))
    all_cosine_similarities.append(cosine_similarity.round(3))

    
    table.add_row(["L2"]+all_l2_distances)
    table.add_row(["Cosine"]+all_cosine_similarities)
    print(table)
    
    
def main(args):
    # load data
    pairs_data = json.load(open(os.path.join(args.raw_data_dir, 'query_ground_truth_pairs.json'), 'r'))
    test_pairs, queries, corpus = load_synthetic_dataset(data_dir=args.raw_data_dir, split=args.split, normalize=args.normalize, indexed_corpus=args.indexed_corpus)

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
                    dataloader = load_input_data(args.embedding_data_dir, use_ground_truth_for_eval=False)
                    
                    # run retrieval
                    all_outputs, _, _, _ = evaluate_loop(dataloader, model, device, max_new_tokens=max_new_tokens, use_gt_q_embed=False, use_eos=False, compute_loss=False)

                    # Evaluate Results
                    rankings = evaluate_baseline_with_aggregation(model_path+f'_max_new_tokens_{max_new_tokens}', all_outputs, corpus, pairs_data[args.split], args.k_values, max_new_tokens)
                    results = eval_metrics(rankings, pairs_data[args.split], args.k_values)
                    _, scores = eval_on_each_gt(rankings, test_pairs, args.k_values, _print=True, num_gt=args.num_gt)
                    
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
    score_results.to_csv('recall_per_gt.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", type=str, nargs='+', default=['results/gaussian_synthetic_inf/gaussian_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep30_warmup0.05/'])
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--raw_data_dir", "-r", type=str, default='./data_creation/gaussian/data/opposing_pairs_data_large/')
    parser.add_argument("--k_values", type=int, nargs='+', default=[1, 5, 10, 20, 50, 100, 500]) # 10, 20, 50, 100, 200, 500
    parser.add_argument("--checkpoint_name", "-c", type=str, default='best_model')
    parser.add_argument("--max_new_tokens_list", "-n", type=int, nargs='+', default=[5])
    parser.add_argument("--normalize", action='store_true')
    parser.add_argument("--indexed_corpus", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="EmbeddingModel")
    parser.add_argument("--run_random_baseline", action='store_true')
    parser.add_argument("--full_finetuning", action='store_true')
    parser.add_argument("--embedding_model_dim", type=int, default=1024)
    parser.add_argument("--embedding_data_dir", type=str, default='./training_datasets/llama-1b/gaussian_linear/inf/gaussian_linear_train_dataset_1b_contrastive/')
    parser.add_argument("--num_gt", type=int, default=5)
    args = parser.parse_args()
    main(args)





