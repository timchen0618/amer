#!/usr/bin/env python3
"""
Baseline Evaluation for Synthetic Information Retrieval Dataset

This script evaluates multiple baseline approaches:
1. Average Baseline: Predicts the average of ground truth vectors
2. Query Baseline: Uses the query vector itself as prediction

Computes Recall@k and MRecall@k metrics for both approaches.
"""

import numpy as np
import json
import os
import argparse
from typing import Dict, List, Tuple, Any
from eval_utils import compute_similarities_and_rankings, compute_recall_at_k, compute_mrecall_at_k
import prettytable

def load_synthetic_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Load synthetic dataset from directory.
    
    Args:
        data_dir: Directory containing the synthetic data files
        
    Returns:
        Tuple of (queries, corpus, query_ground_truth_pairs)
    """
    print(f"Loading data from {data_dir}...")
    
    # Load corpus and queries
    corpus = np.load(os.path.join(data_dir, 'corpus.npy'))
    queries = np.load(os.path.join(data_dir, 'queries.npy'))
    
    # Load query-ground truth pairs
    with open(os.path.join(data_dir, 'query_ground_truth_pairs.json'), 'r') as f:
        pairs_data = json.load(f)
    
    print(f"  Corpus shape: {corpus.shape}")
    print(f"  Queries shape: {queries.shape}")
    print(f"  Train pairs: {len(pairs_data['train'])}")
    print(f"  Test pairs: {len(pairs_data['test'])}")
    
    return queries, corpus, pairs_data


def compute_average_baseline_predictions(queries: np.ndarray, corpus: np.ndarray, 
                                       pairs_data: Dict[str, Any]) -> np.ndarray:
    """
    Compute baseline predictions by averaging the 5 ground truth vectors for each query.
    
    Args:
        queries: Query vectors
        corpus: Full corpus vectors  
        pairs_data: Query-ground truth mapping data
        
    Returns:
        Array of predicted vectors for test queries
    """
    print("Computing average baseline predictions...")
    
    test_pairs = pairs_data['test']
    predictions = []
    
    for pair in test_pairs:
        gt_indices = pair['ground_truth_indices']
        
        if len(gt_indices) == 0:
            print(f"Warning: Query {pair['query_idx']} has zero ground truth vectors!")
            # Use query vector as fallback
            predictions.append(queries[pair['query_idx']])
            continue
        
        # Get ground truth vectors from corpus
        gt_vectors = corpus[gt_indices]
        
        # Compute average
        avg_prediction = np.mean(gt_vectors, axis=0)
        predictions.append(avg_prediction)
    
    predictions = np.array(predictions)
    print(f"  Generated {len(predictions)} average baseline predictions")
    return predictions


def compute_second_gt_predictions(queries: np.ndarray, corpus: np.ndarray, 
                                       pairs_data: Dict[str, Any]) -> np.ndarray:
    print("Computing second ground truth predictions...")
    
    test_pairs = pairs_data['test']
    predictions = []
    
    for pair in test_pairs:
        gt_index = pair['ground_truth_indices'][1]
        predictions.append(corpus[gt_index])
    
    predictions = np.array(predictions)
    print(f"  Generated {len(predictions)} second ground truth predictions")
    return predictions

def compute_average_2nd_3rd_gt_baseline_predictions(queries: np.ndarray, corpus: np.ndarray, 
                                       pairs_data: Dict[str, Any]) -> np.ndarray:
    print("Computing average 2nd and 3rd ground truth predictions...")
    
    test_pairs = pairs_data['test']
    predictions = []
    
    for pair in test_pairs:
        gt_indices = pair['ground_truth_indices'][1:3]
        
        if len(gt_indices) == 0:
            print(f"Warning: Query {pair['query_idx']} has zero ground truth vectors!")
            # Use query vector as fallback
            predictions.append(queries[pair['query_idx']])
            continue
        
        # Get ground truth vectors from corpus
        gt_vectors = corpus[gt_indices]
        
        # Compute average
        avg_prediction = np.mean(gt_vectors, axis=0)
        predictions.append(avg_prediction)
    
    predictions = np.array(predictions)
    print(f"  Generated {len(predictions)} average 2nd and 3rd ground truth predictions")
    return predictions

def compute_average_1st_2nd_3rd_gt_baseline_predictions(queries: np.ndarray, corpus: np.ndarray, 
                                       pairs_data: Dict[str, Any]) -> np.ndarray:
    print("Computing average 1st, 2nd and 3rd ground truth predictions...")
    
    test_pairs = pairs_data['test']
    predictions = []

    for pair in test_pairs:
        gt_indices = pair['ground_truth_indices'][0:3]
        
        if len(gt_indices) == 0:
            print(f"Warning: Query {pair['query_idx']} has zero ground truth vectors!")
            # Use query vector as fallback
            predictions.append(queries[pair['query_idx']])
            continue
        
        # Get ground truth vectors from corpus
        gt_vectors = corpus[gt_indices]
        
        # Compute average
        avg_prediction = np.mean(gt_vectors, axis=0)
        predictions.append(avg_prediction)
    
    predictions = np.array(predictions)
    print(f"  Generated {len(predictions)} average 1st, 2nd and 3rd ground truth predictions")
    return predictions

def compute_average_1st_2nd_3rd_4th_gt_baseline_predictions(queries: np.ndarray, corpus: np.ndarray, 
                                       pairs_data: Dict[str, Any]) -> np.ndarray:
    print("Computing average 1st, 2nd, 3rd and 4th ground truth predictions...")
    
    test_pairs = pairs_data['test']
    predictions = []

    for pair in test_pairs:
        gt_indices = pair['ground_truth_indices'][0:4]
        
        if len(gt_indices) == 0:
            print(f"Warning: Query {pair['query_idx']} has zero ground truth vectors!")
            # Use query vector as fallback
            predictions.append(queries[pair['query_idx']])
            continue
        
        # Get ground truth vectors from corpus
        gt_vectors = corpus[gt_indices]
        
        # Compute average
        avg_prediction = np.mean(gt_vectors, axis=0)
        predictions.append(avg_prediction)
    
    predictions = np.array(predictions)
    print('predictions.shape', predictions.shape)
    print(f"  Generated {len(predictions)} average 1st, 2nd, 3rd and 4th ground truth predictions")
    return predictions

def compute_any_two_or_three_gt_baseline_predictions(queries: np.ndarray, corpus: np.ndarray, 
                                       pairs_data: Dict[str, Any], k_values: List[int]) -> np.ndarray:
    # select 3 random ground truth indices
    # iterate over all combinations of 3 indices
    # compute the average of the vectors at the indices
    # append the average to the predictions
    import itertools
    # Define the set of 5 elements
    elements = [0, 1, 2, 3, 4]
    for combs in [2, 3, 4]:
        print('0'*100)
        # Generate combinations of 3 elements from the set
        combinations = list(itertools.combinations(elements, combs))
        
        all_results = []
        for combination in combinations:
            test_pairs = pairs_data['test']
            predictions = []

            for pair in test_pairs:
                gt_indices = [pair['ground_truth_indices'][c] for c in combination]
                if len(gt_indices) == 0:
                    print(f"Warning: Query {pair['query_idx']} has zero ground truth vectors!")
                    # Use query vector as fallback
                    predictions.append(queries[pair['query_idx']])
                    continue
                
                # Get ground truth vectors from corpus
                gt_vectors = corpus[gt_indices]
                
                # Compute average
                avg_prediction = np.mean(gt_vectors, axis=0)
                predictions.append(avg_prediction)
            
            predictions = np.array(predictions)
            print('predictions.shape', predictions.shape)
            results = evaluate_baseline(f"Average  Ground Truth Baseline", predictions, corpus, pairs_data['test'], k_values)
            all_results.append((combination, results))
            
        # Create a table to display the results
        table = prettytable.PrettyTable()
        table.field_names = ["Combination", "Recall@5", "Recall@10", "Recall@20", "MRecall@5", "MRecall@10", "MRecall@20"]
        
        # Add rows to the table 
        for combination, results in all_results:
            table.add_row([combination, '%2.2f' % results['recall@5'], '%2.2f' % results['recall@10'], '%2.2f' % results['recall@20'], '%2.2f' % results['mrecall@5'], '%2.2f' % results['mrecall@10'], '%2.2f' % results['mrecall@20']])
        
        print(table)
    
    

    

def compute_query_baseline_predictions(queries: np.ndarray, 
                                     pairs_data: Dict[str, Any]) -> np.ndarray:
    """
    Compute baseline predictions using query vectors themselves.
    
    Args:
        queries: Query vectors
        pairs_data: Query-ground truth mapping data
        
    Returns:
        Array of predicted vectors for test queries (same as query vectors)
    """
    print("Computing query baseline predictions...")
    
    test_pairs = pairs_data['test']
    predictions = []
    
    for pair in test_pairs:
        query_idx = pair['query_idx']
        # Use the query vector itself as prediction
        predictions.append(queries[query_idx])
    
    predictions = np.array(predictions)
    print(f"  Generated {len(predictions)} query baseline predictions")
    return predictions





def evaluate_baseline(baseline_name: str, predictions: np.ndarray, corpus: np.ndarray, 
                     test_pairs: List[Dict[str, Any]], k_values: List[int]) -> Dict[str, float]:
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
    similarities, rankings = compute_similarities_and_rankings(predictions, corpus, max_k=max(k_values), _print=False)
    
    results = {}
    
    # Evaluate for each k
    for k in k_values:
        recall = compute_recall_at_k(rankings, test_pairs, k)
        mrecall = compute_mrecall_at_k(rankings, test_pairs, k)
        
        results[f'recall@{k}'] = recall
        results[f'mrecall@{k}'] = mrecall
        
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  MRecall@{k}: {mrecall:.4f}")
    
    return results


def analyze_ground_truth_distribution(pairs_data: Dict[str, Any]):
    """Analyze the distribution of ground truth vectors per query."""
    print("\n=== Ground Truth Distribution Analysis ===")
    
    test_pairs = pairs_data['test']
    gt_counts = [len(pair['ground_truth_indices']) for pair in test_pairs]
    
    print(f"Ground truth vectors per query:")
    print(f"  Min: {min(gt_counts)}")
    print(f"  Max: {max(gt_counts)}")
    print(f"  Mean: {np.mean(gt_counts):.2f}")
    print(f"  Std: {np.std(gt_counts):.2f}")
    
    # Count queries by number of ground truth vectors
    from collections import Counter
    gt_distribution = Counter(gt_counts)
    print(f"  Distribution: {dict(gt_distribution)}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate multiple baselines for synthetic IR dataset')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Directory containing synthetic data')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20],
                       help='k values for Recall@k and MRecall@k (default: 5 10 20)')
    
    args = parser.parse_args()
    
    # Load data
    queries, corpus, pairs_data = load_synthetic_data(args.data_dir)
    
    # Analyze ground truth distribution
    analyze_ground_truth_distribution(pairs_data)
    
    # Compute predictions for both baselines
    avg_predictions = compute_average_baseline_predictions(queries, corpus, pairs_data)
    query_predictions = compute_query_baseline_predictions(queries, pairs_data)
    second_gt_predictions = compute_second_gt_predictions(queries, corpus, pairs_data)
    avg_2nd_3rd_gt_predictions = compute_average_2nd_3rd_gt_baseline_predictions(queries, corpus, pairs_data)
    avg_1st_2nd_3rd_gt_predictions = compute_average_1st_2nd_3rd_gt_baseline_predictions(queries, corpus, pairs_data)
    avg_1st_2nd_3rd_4th_gt_predictions = compute_average_1st_2nd_3rd_4th_gt_baseline_predictions(queries, corpus, pairs_data)
    print('avg_predictions.shape', avg_predictions.shape)
    print('query_predictions.shape', query_predictions.shape)
    print('second_gt_predictions.shape', second_gt_predictions.shape)
    print('avg_2nd_3rd_gt_predictions.shape', avg_2nd_3rd_gt_predictions.shape)
    print('avg_1st_2nd_3rd_gt_predictions.shape', avg_1st_2nd_3rd_gt_predictions.shape)
    print('avg_1st_2nd_3rd_4th_gt_predictions.shape', avg_1st_2nd_3rd_4th_gt_predictions.shape) 
    # Evaluate both baselines
    avg_results = evaluate_baseline("Average Baseline", avg_predictions, corpus, pairs_data['test'], args.k_values)
    query_results = evaluate_baseline("Query Baseline", query_predictions, corpus, pairs_data['test'], args.k_values)
    second_gt_results = evaluate_baseline("Second Ground Truth Baseline", second_gt_predictions, corpus, pairs_data['test'], args.k_values)
    avg_2nd_3rd_gt_results = evaluate_baseline("Average 2nd and 3rd Ground Truth Baseline", avg_2nd_3rd_gt_predictions, corpus, pairs_data['test'], args.k_values)
    avg_1st_2nd_3rd_gt_results = evaluate_baseline("Average 1st, 2nd and 3rd Ground Truth Baseline", avg_1st_2nd_3rd_gt_predictions, corpus, pairs_data['test'], args.k_values)
    avg_1st_2nd_3rd_4th_gt_results = evaluate_baseline("Average 1st, 2nd, 3rd and 4th Ground Truth Baseline", avg_1st_2nd_3rd_4th_gt_predictions, corpus, pairs_data['test'], args.k_values)
    # Print comparison table
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    print(f"{'Metric':<15} {'Average':<12} {'Query':<12} {'Second GT':<12} {'Avg 2 and 3 GT':<12} {'Avg 1, 2 and 3 GT':<12} {'Avg 1, 2, 3 and 4 GT':<12}")
    print("-" * 60)
    
    for k in args.k_values:
        recall_key = f'recall@{k}'
        mrecall_key = f'mrecall@{k}'
        
        avg_recall = avg_results[recall_key]
        query_recall = query_results[recall_key]
        second_gt_recall = second_gt_results[recall_key]
        avg_2nd_3rd_gt_recall = avg_2nd_3rd_gt_results[recall_key]
        avg_1st_2nd_3rd_gt_recall = avg_1st_2nd_3rd_gt_results[recall_key]
        avg_1st_2nd_3rd_4th_gt_recall = avg_1st_2nd_3rd_4th_gt_results[recall_key]
        recall_diff = avg_recall - query_recall
        
        avg_mrecall = avg_results[mrecall_key]
        query_mrecall = query_results[mrecall_key]
        second_gt_mrecall = second_gt_results[mrecall_key]
        avg_2nd_3rd_gt_mrecall = avg_2nd_3rd_gt_results[mrecall_key]
        avg_1st_2nd_3rd_gt_mrecall = avg_1st_2nd_3rd_gt_results[mrecall_key]
        avg_1st_2nd_3rd_4th_gt_mrecall = avg_1st_2nd_3rd_4th_gt_results[mrecall_key]
        mrecall_diff = avg_mrecall - query_mrecall
        
        print(f"{recall_key:<15} {avg_recall:<12.4f} {query_recall:<12.4f} {second_gt_recall:<12.4f} {avg_2nd_3rd_gt_recall:<12.4f} {avg_1st_2nd_3rd_gt_recall:<12.4f} {avg_1st_2nd_3rd_4th_gt_recall:<12.4f}")
        print(f"{mrecall_key:<15} {avg_mrecall:<12.4f} {query_mrecall:<12.4f} {second_gt_mrecall:<12.4f} {avg_2nd_3rd_gt_mrecall:<12.4f} {avg_1st_2nd_3rd_gt_mrecall:<12.4f} {avg_1st_2nd_3rd_4th_gt_mrecall:<12.4f}")
    
    print("-" * 60)
    
    # Determine which baseline performs better
    avg_score = np.mean([avg_results[f'recall@{k}'] for k in args.k_values])
    query_score = np.mean([query_results[f'recall@{k}'] for k in args.k_values])

    print(f"\nOverall Performance:")
    print(f"  Average Baseline: {avg_score:.4f}")
    print(f"  Query Baseline: {query_score:.4f}")
    
    if query_score > avg_score:
        print(f"  🚨 WARNING: Query baseline outperforms averaging! Dataset might be too easy.")
    elif avg_score > 0.8:
        print(f"  ⚠️  CAUTION: Both baselines score very high. Dataset might be too easy.")
    else:
        print(f"  ✅ Good: Both baselines have reasonable difficulty.")

    print('-'*100)
    print('Computing any three ground truth predictions...')
    print('-'*100)
    compute_any_two_or_three_gt_baseline_predictions(queries, corpus, pairs_data, args.k_values)

if __name__ == '__main__':
    main() 