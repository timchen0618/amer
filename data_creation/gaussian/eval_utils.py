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


def compute_similarities_and_rankings(predictions: np.ndarray, corpus: np.ndarray, max_k: int, _print: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarities between predictions and corpus, then rank by similarity.
    
    Args:
        predictions: Predicted vectors for test queries
        corpus: Full corpus to search in
        
    Returns:
        Tuple of (similarities, rankings) where rankings[i] gives corpus indices sorted by similarity to prediction i
    """
    if _print:
        print("Computing similarities and rankings...")
    batch_size = 100000
    if corpus.shape[0] <= batch_size:
        # Compute cosine similarities between all predictions and all corpus vectors
        similarities = cosine_similarity(predictions, corpus)
        if _print:
            print(np.sort(similarities, axis=1)[:, ::-1][:, :10])
        
        # For each prediction, get corpus indices ranked by similarity (descending)
        rankings = np.argsort(-similarities, axis=1)  # Negative for descending order
        
        if _print:
            print(f"  Computed similarities: {similarities.shape}")
            # print(similarities[:, :10])
            print(rankings[:, :10])
        return similarities, rankings
    else:
        top_k_similarities_batch = []
        rankings_batch = []
        for i in range(0, corpus.shape[0], batch_size):
            similarity_batch = cosine_similarity(predictions, corpus[i:i+batch_size])
            sorted_indices = np.argsort(similarity_batch, axis=1)[:, ::-1][:, :max_k]
            sorted_similarity_batch = similarity_batch[np.arange(similarity_batch.shape[0])[:, None], sorted_indices]
            rankings_batch.append(sorted_indices+i)
            top_k_similarities_batch.append(sorted_similarity_batch)
        # print(i, len(top_k_similarities_batch), len(rankings_batch),top_k_similarities_batch[0].shape, rankings_batch[0].shape)
        similarities = np.concatenate(top_k_similarities_batch, axis=1)
        rankings = np.concatenate(rankings_batch, axis=1)
        # print('similarities', similarities.shape, 'rankings', rankings.shape)
        
        sorted_indices = np.argsort(similarities, axis=1)[:, ::-1]
        sorted_rankings = rankings[np.arange(rankings.shape[0])[:, None], sorted_indices]
        sorted_similarities = similarities[np.arange(similarities.shape[0])[:, None], sorted_indices]
        # print('sorted_indices', sorted_indices.shape, 'sorted_rankings', sorted_rankings.shape, 'sorted_similarities', sorted_similarities.shape)
        if _print:
            print(sorted_similarities[:, :10])
            print(sorted_rankings[:, :10])
            print(f"  Computed similarities: {sorted_similarities.shape}")
        return sorted_similarities, sorted_rankings