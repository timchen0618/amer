#!/usr/bin/env python3
"""
Baseline evaluation script for synthetic information retrieval dataset.
Computes baseline performance using average of ground truth vectors as predictions.
"""

import numpy as np
import json
import os
import argparse
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def load_synthetic_data(data_dir: str) -> Dict:
    """
    Load the generated synthetic data from files.
    
    Args:
        data_dir: Directory containing the synthetic data files
        
    Returns:
        Dictionary containing all loaded data
    """
    print(f"Loading synthetic data from {data_dir}...")
    
    try:
        # Load configuration
        with open(os.path.join(data_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Load data files
        corpus = np.load(os.path.join(data_dir, 'corpus.npy'))
        queries = np.load(os.path.join(data_dir, 'queries.npy'))
        
        # Load query-ground truth pairs
        with open(os.path.join(data_dir, 'query_ground_truth_pairs.json'), 'r') as f:
            pairs_data = json.load(f)
        
        print("✓ All files loaded successfully")
        
        return {
            'config': config,
            'corpus': corpus,
            'queries': queries,
            'pairs_data': pairs_data
        }
        
    except FileNotFoundError as e:
        print(f"❌ Error: Missing file - {e}")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


def compute_baseline_predictions(corpus: np.ndarray, pairs_data: Dict) -> Dict[int, np.ndarray]:
    """
    Compute baseline predictions by averaging the ground truth vectors for each query.
    
    Args:
        corpus: Corpus vectors
        pairs_data: Query-ground truth mappings
        
    Returns:
        Dictionary mapping query_idx to predicted vector (average of ground truth)
    """
    print("Computing baseline predictions (average of ground truth vectors)...")
    
    predictions = {}
    all_pairs = pairs_data['train'] + pairs_data['test']
    
    for pair in all_pairs:
        query_idx = pair['query_idx']
        gt_indices = pair['ground_truth_indices']
        
        # Get ground truth vectors
        gt_vectors = corpus[gt_indices]
        
        # Compute average
        avg_vector = np.mean(gt_vectors, axis=0)
        predictions[query_idx] = avg_vector
    
    print(f"Computed predictions for {len(predictions)} queries")
    return predictions


def compute_similarities_and_rankings(predictions: Dict[int, np.ndarray], 
                                    corpus: np.ndarray, 
                                    query_indices: List[int]) -> Dict[int, np.ndarray]:
    """
    Compute cosine similarities between predictions and corpus, return rankings.
    
    Args:
        predictions: Dictionary of predicted vectors
        corpus: Corpus vectors
        query_indices: List of query indices to evaluate
        
    Returns:
        Dictionary mapping query_idx to ranked corpus indices (descending similarity)
    """
    print("Computing similarities and rankings...")
    
    rankings = {}
    
    for query_idx in tqdm(query_indices):
        pred_vector = predictions[query_idx].reshape(1, -1)
        
        # Compute cosine similarity with entire corpus
        similarities = cosine_similarity(pred_vector, corpus)[0]
        
        # Get ranking (indices sorted by similarity, descending)
        ranking = np.argsort(similarities)[::-1]
        rankings[query_idx] = ranking
    
    return rankings


def compute_recall_at_k(rankings: Dict[int, np.ndarray], 
                       pairs_data: Dict, 
                       k: int, 
                       split: str = 'test') -> float:
    """
    Compute standard Recall@k.
    
    Args:
        rankings: Dictionary of ranked corpus indices
        pairs_data: Query-ground truth mappings
        k: Number of top results to consider
        split: 'train' or 'test'
        
    Returns:
        Recall@k score
    """
    pairs = pairs_data[split]
    total_relevant = 0
    total_retrieved_relevant = 0
    
    for pair in pairs:
        query_idx = pair['query_idx']
        gt_indices = set(pair['ground_truth_indices'])
        
        # Get top-k predictions
        top_k = set(rankings[query_idx][:k])
        
        # Count relevant items retrieved
        retrieved_relevant = len(gt_indices.intersection(top_k))
        
        total_relevant += len(gt_indices)
        total_retrieved_relevant += retrieved_relevant
    
    recall = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0.0
    return recall


def compute_mrecall_at_k(rankings: Dict[int, np.ndarray], 
                        pairs_data: Dict, 
                        k: int, 
                        split: str = 'test') -> float:
    """
    Compute MRecall@k as defined:
    - If number of ground truth > k: check if k different ground truth vectors are retrieved
    - If number of ground truth <= k: check if all ground truth vectors are retrieved
    
    Args:
        rankings: Dictionary of ranked corpus indices
        pairs_data: Query-ground truth mappings
        k: Number of top results to consider
        split: 'train' or 'test'
        
    Returns:
        MRecall@k score
    """
    pairs = pairs_data[split]
    successful_queries = 0
    total_queries = len(pairs)
    
    for pair in pairs:
        query_idx = pair['query_idx']
        gt_indices = set(pair['ground_truth_indices'])
        num_gt = len(gt_indices)
        
        # Get top-k predictions
        top_k = set(rankings[query_idx][:k])
        
        # Count relevant items retrieved
        retrieved_relevant = len(gt_indices.intersection(top_k))
        
        # Apply MRecall@k logic
        if num_gt > k:
            # Check if k different ground truth vectors are retrieved
            if retrieved_relevant >= k:
                successful_queries += 1
        else:
            # Check if all ground truth vectors are retrieved
            if retrieved_relevant == num_gt:
                successful_queries += 1
    
    mrecall = successful_queries / total_queries if total_queries > 0 else 0.0
    return mrecall


def evaluate_baseline(data_dir: str, k_values: List[int] = [5, 10, 20]):
    """
    Run complete baseline evaluation.
    
    Args:
        data_dir: Directory containing synthetic data
        k_values: List of k values to evaluate
    """
    print("=== Baseline Evaluation: Average Ground Truth Prediction ===\n")
    
    # Load data
    data = load_synthetic_data(data_dir)
    
    # Compute baseline predictions
    predictions = compute_baseline_predictions(data['corpus'], data['pairs_data'])
    
    # Get test query indices
    test_pairs = data['pairs_data']['test']
    test_query_indices = [pair['query_idx'] for pair in test_pairs]
    
    # Compute rankings
    rankings = compute_similarities_and_rankings(predictions, data['corpus'], test_query_indices)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Evaluated on {len(test_query_indices)} test queries")
    print(f"Corpus size: {len(data['corpus'])}")
    if 'n_ground_truth_per_query' in data['pairs_data']['metadata']:
        gt_per_query = data['pairs_data']['metadata']['n_ground_truth_per_query']
    else:
        gt_per_query = data['pairs_data']['metadata']['n_rotations']
    print(f"Ground truth vectors per query: {gt_per_query}")
    
    print(f"\n{'Metric':<15} {'k=5':<10} {'k=10':<10} {'k=20':<10}")
    print("-" * 50)
    
    # Compute metrics for each k
    recall_results = []
    mrecall_results = []
    
    for k in k_values:
        recall_k = compute_recall_at_k(rankings, data['pairs_data'], k, 'test')
        mrecall_k = compute_mrecall_at_k(rankings, data['pairs_data'], k, 'test')
        
        recall_results.append(recall_k)
        mrecall_results.append(mrecall_k)
    
    # Print results
    recall_str = f"{'Recall@k':<15}"
    mrecall_str = f"{'MRecall@k':<15}"
    
    for i, k in enumerate(k_values):
        recall_str += f"{recall_results[i]:<10.4f}"
        mrecall_str += f"{mrecall_results[i]:<10.4f}"
    
    print(recall_str)
    print(mrecall_str)
    
    # Detailed analysis
    print(f"\n=== Detailed Analysis ===")
    
    # Analyze ground truth distribution
    gt_counts = {}
    for pair in test_pairs:
        num_gt = len(pair['ground_truth_indices'])
        gt_counts[num_gt] = gt_counts.get(num_gt, 0) + 1
    
    print(f"Ground truth distribution:")
    for num_gt, count in sorted(gt_counts.items()):
        print(f"  {num_gt} ground truth vectors: {count} queries")
    
    # Per-query analysis for k=5
    print(f"\nMRecall@5 analysis:")
    successful_5 = 0
    for pair in test_pairs:
        query_idx = pair['query_idx']
        gt_indices = set(pair['ground_truth_indices'])
        num_gt = len(gt_indices)
        
        top_5 = set(rankings[query_idx][:5])
        retrieved_relevant = len(gt_indices.intersection(top_5))
        
        if num_gt > 5:
            success = retrieved_relevant >= 5
        else:
            success = retrieved_relevant == num_gt
            
        if success:
            successful_5 += 1
    
    print(f"  Successful queries: {successful_5}/{len(test_pairs)} ({successful_5/len(test_pairs)*100:.1f}%)")
    
    return {
        'recall': {f'k_{k}': recall_results[i] for i, k in enumerate(k_values)},
        'mrecall': {f'k_{k}': mrecall_results[i] for i, k in enumerate(k_values)},
        'num_test_queries': len(test_query_indices),
        'corpus_size': len(data['corpus']),
        'gt_per_query': gt_per_query
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline performance on synthetic IR data')
    parser.add_argument('--data-dir', '-d', type=str, default='./synthetic_data',
                       help='Directory containing synthetic data (default: ./synthetic_data)')
    parser.add_argument('--k-values', nargs='+', type=int, default=[5, 10, 20],
                       help='k values for evaluation (default: 5 10 20)')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory '{args.data_dir}' not found!")
        print("Please run 'python generate_data.py' first to generate the data.")
        return
    
    try:
        results = evaluate_baseline(args.data_dir, args.k_values)
        print(f"\n✅ Baseline evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 