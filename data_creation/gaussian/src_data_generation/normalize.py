#!/usr/bin/env python3
import numpy as np
import json
import os
import argparse
from typing import Dict, List, Tuple, Any
from sklearn.preprocessing import normalize


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
    
    print(f"  Corpus shape: {corpus.shape}")
    print(f"  Queries shape: {queries.shape}")    
    return queries, corpus


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize each query vector to have unit norm.
    
    Args:
        queries: numpy array of shape (n, d)
        
    Returns:
        numpy array of shape (n, d)
    """
    return normalize(vectors, axis=1, norm='l2')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    queries, corpus = load_synthetic_data(args.data_dir)
    normalized_queries = normalize_vectors(queries)
    normalized_corpus = normalize_vectors(corpus)
    # l2_norm_queries = np.linalg.norm(normalized_queries, axis=1, keepdims=True)
    # l2_norm_corpus = np.linalg.norm(normalized_corpus, axis=1, keepdims=True)
    # print('l2_norm_queries', l2_norm_queries, l2_norm_queries.shape)
    # print('l2_norm_corpus', l2_norm_corpus, l2_norm_corpus.shape)

    np.save(os.path.join(args.data_dir, 'normalized_queries.npy'), normalized_queries)
    np.save(os.path.join(args.data_dir, 'normalized_corpus.npy'), normalized_corpus)


if __name__ == '__main__':
    main()