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
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any


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


def compute_similarities_and_rankings(predictions: np.ndarray, corpus: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarities between predictions and corpus, then rank by similarity.
    
    Args:
        predictions: Predicted vectors for test queries
        corpus: Full corpus to search in
        
    Returns:
        Tuple of (similarities, rankings) where rankings[i] gives corpus indices sorted by similarity to prediction i
    """
    print("Computing similarities and rankings...")
    
    # Compute cosine similarities between all predictions and all corpus vectors
    similarities = cosine_similarity(predictions, corpus)
    
    # For each prediction, get corpus indices ranked by similarity (descending)
    rankings = np.argsort(-similarities, axis=1)  # Negative for descending order
    
    print(f"  Computed similarities: {similarities.shape}")
    return similarities, rankings


def compute_recall_at_k(rankings: np.ndarray, pairs_data: Dict[str, Any], k: int) -> float:
    """
    Compute Recall@k: percentage of ground truth vectors found in top-k results.
    Uses macro averaging - computes recall per query and averages the results.
    
    Args:
        rankings: Ranked corpus indices for each test query
        pairs_data: Query-ground truth mapping data
        k: Number of top results to consider
        
    Returns:
        Macro-averaged Recall@k score
    """
    test_pairs = pairs_data['test']
    query_recalls = []
    
    for i, pair in enumerate(test_pairs):
        gt_indices = set(pair['ground_truth_indices'])
        top_k_indices = set(rankings[i, :k])
        
        # Compute recall for this query
        gt_found = len(gt_indices.intersection(top_k_indices))
        query_recall = gt_found / len(gt_indices)
        query_recalls.append(query_recall)
    
    # Macro average - mean of per-query recalls
    macro_recall = np.mean(query_recalls)
    return 100*macro_recall


def compute_mrecall_at_k(rankings: np.ndarray, pairs_data: Dict[str, Any], k: int) -> float:
    """
    Compute Modified Recall@k (MRecall@k):
    - If |GT| >= k: success if at least k ground truth vectors in top-k
    - If |GT| < k: success if all ground truth vectors in top-k
    
    Args:
        rankings: Ranked corpus indices for each test query  
        pairs_data: Query-ground truth mapping data
        k: Number of top results to consider
        
    Returns:
        MRecall@k score
    """
    test_pairs = pairs_data['test']
    successful_queries = 0
    
    for i, pair in enumerate(test_pairs):
        gt_indices = set(pair['ground_truth_indices'])
        top_k_indices = set(rankings[i, :k])
        
        gt_in_top_k = len(gt_indices.intersection(top_k_indices))
        
        if len(gt_indices) >= k:
            # Need at least k ground truth in top-k
            success = gt_in_top_k >= k
        else:
            # Need all ground truth in top-k
            success = gt_in_top_k == len(gt_indices)
        
        if success:
            successful_queries += 1
    
    mrecall = successful_queries / len(test_pairs)
    return 100*mrecall


def evaluate_baseline(baseline_name: str, predictions: np.ndarray, corpus: np.ndarray, 
                     pairs_data: Dict[str, Any], k_values: List[int]) -> Dict[str, float]:
    """
    Evaluate a baseline approach with multiple k values.
    
    Args:
        baseline_name: Name of the baseline approach
        predictions: Predicted vectors for test queries
        corpus: Full corpus to search in
        pairs_data: Query-ground truth mapping data
        k_values: List of k values to evaluate
        
    Returns:
        Dictionary of metric results
    """
    print(f"\n=== Evaluating {baseline_name} ===")
    
    # Compute similarities and rankings
    similarities, rankings = compute_similarities_and_rankings(predictions, corpus)
    
    results = {}
    
    # Evaluate for each k
    for k in k_values:
        recall = compute_recall_at_k(rankings, pairs_data, k)
        mrecall = compute_mrecall_at_k(rankings, pairs_data, k)
        
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
    print('avg_predictions.shape', avg_predictions.shape)
    print('query_predictions.shape', query_predictions.shape)
    
    # Evaluate both baselines
    avg_results = evaluate_baseline("Average Baseline", avg_predictions, corpus, pairs_data, args.k_values)
    query_results = evaluate_baseline("Query Baseline", query_predictions, corpus, pairs_data, args.k_values)
    
    # Print comparison table
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    print(f"{'Metric':<15} {'Average':<12} {'Query':<12} {'Difference':<12}")
    print("-" * 60)
    
    for k in args.k_values:
        recall_key = f'recall@{k}'
        mrecall_key = f'mrecall@{k}'
        
        avg_recall = avg_results[recall_key]
        query_recall = query_results[recall_key]
        recall_diff = avg_recall - query_recall
        
        avg_mrecall = avg_results[mrecall_key]
        query_mrecall = query_results[mrecall_key]
        mrecall_diff = avg_mrecall - query_mrecall
        
        print(f"{recall_key:<15} {avg_recall:<12.4f} {query_recall:<12.4f} {recall_diff:+.4f}")
        print(f"{mrecall_key:<15} {avg_mrecall:<12.4f} {query_mrecall:<12.4f} {mrecall_diff:+.4f}")
    
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


if __name__ == '__main__':
    main() 