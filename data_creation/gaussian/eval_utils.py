from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any
import numpy as np
import json


def compute_recall_at_k(rankings: np.ndarray, test_pairs: List[Dict[str, Any]], k: int) -> float:
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
    # test_pairs = pairs_data['test']
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


def compute_mrecall_at_k(rankings: np.ndarray, test_pairs: List[Dict[str, Any]], k: int) -> float:
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
    # test_pairs = pairs_data['test']
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
    print(np.sort(similarities, axis=1)[:, ::-1][:, :10])
    
    # For each prediction, get corpus indices ranked by similarity (descending)
    rankings = np.argsort(-similarities, axis=1)  # Negative for descending order
    
    
    print(f"  Computed similarities: {similarities.shape}")
    return similarities, rankings