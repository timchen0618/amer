#!/usr/bin/env python3
"""
Data validation script for synthetic information retrieval dataset.
Tests data integrity and computes distance statistics.
"""

import numpy as np
import json
import os
import argparse
from typing import Dict, List, Tuple
import random


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
        if 'transformation_type' in config and config['transformation_type'] == 'diverse_mlps':
            transformation_matrices = None
        else:
            transformation_matrices = np.load(os.path.join(data_dir, 'transformation_matrices.npy'))
        # ground_truth_indices = np.load(os.path.join(data_dir, 'ground_truth_indices.npy'))
        
        # Load query-ground truth pairs
        with open(os.path.join(data_dir, 'query_ground_truth_pairs.json'), 'r') as f:
            pairs_data = json.load(f)
        
        print("✓ All files loaded successfully")
        
        return {
            'config': config,
            'corpus': corpus,
            'queries': queries,
            'transformation_matrices': transformation_matrices,
            'pairs_data': pairs_data
        }
        
    except FileNotFoundError as e:
        print(f"❌ Error: Missing file - {e}")
        raise
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        raise


def validate_data_integrity(data: Dict, use_mlp_transformations: bool = False) -> bool:
    """
    Validate the integrity and consistency of the loaded data.
    
    Args:
        data: Dictionary containing all loaded data
        
    Returns:
        True if all validations pass, False otherwise
    """
    print("\n=== Data Integrity Validation ===")
    
    config = data['config']
    corpus = data['corpus']
    queries = data['queries']
    if use_mlp_transformations:
        pass
    else:        
        transformation_matrices = data['transformation_matrices']
    # ground_truth_indices = data['ground_truth_indices']
    pairs_data = data['pairs_data']
    
    validation_passed = True
    
    # Check dimensions
    expected_corpus_shape = (config['corpus_size'], config['dimensions'])
    expected_queries_shape = (config['n_train_queries'] + config['n_test_queries'], config['dimensions'])
    expected_rotations_shape = (config['n_transformations'], config['dimensions'], config['dimensions'])
    
    if corpus.shape != expected_corpus_shape:
        print(f"❌ Corpus shape mismatch: expected {expected_corpus_shape}, got {corpus.shape}")
        validation_passed = False
    else:
        print(f"✓ Corpus shape: {corpus.shape}")
    
    if queries.shape != expected_queries_shape:
        print(f"❌ Queries shape mismatch: expected {expected_queries_shape}, got {queries.shape}")
        validation_passed = False
    else:
        print(f"✓ Queries shape: {queries.shape}")
    
    if use_mlp_transformations:
        pass
    else:
        if transformation_matrices.shape != expected_rotations_shape:
            print(f"❌ Rotation matrices shape mismatch: expected {expected_rotations_shape}, got {transformation_matrices.shape}")
            validation_passed = False
        else:
            print(f"✓ Rotation matrices shape: {transformation_matrices.shape}")
    
    # # Check ground truth indices
    # if len(ground_truth_indices) != config['corpus_size']:
    #     print(f"❌ Ground truth indices length mismatch: expected {config['corpus_size']}, got {len(ground_truth_indices)}")
    #     validation_passed = False
    # else:
    #     print(f"✓ Ground truth indices length: {len(ground_truth_indices)}")
    
    # # Check number of ground truth vectors
    # expected_gt_count = (config['n_train_queries'] + config['n_test_queries']) * config['n_transformations']
    # actual_gt_count = np.sum(ground_truth_indices)
    # if actual_gt_count != expected_gt_count:
    #     print(f"❌ Ground truth count mismatch: expected {expected_gt_count}, got {actual_gt_count}")
    #     validation_passed = False
    # else:
    #     print(f"✓ Ground truth vectors count: {actual_gt_count}")
    
    # Check train/test split
    if len(pairs_data['train']) != config['n_train_queries']:
        print(f"❌ Training pairs count mismatch: expected {config['n_train_queries']}, got {len(pairs_data['train'])}")
        validation_passed = False
    else:
        print(f"✓ Training pairs count: {len(pairs_data['train'])}")
    
    if len(pairs_data['test']) != config['n_test_queries']:
        print(f"❌ Test pairs count mismatch: expected {config['n_test_queries']}, got {len(pairs_data['test'])}")
        validation_passed = False
    else:
        print(f"✓ Test pairs count: {len(pairs_data['test'])}")
    
    # Check that each query has the expected number of ground truth vectors
    expected_gt_per_query = config['n_transformations']
    for split_name, pairs in [('train', pairs_data['train']), ('test', pairs_data['test'])]:
        for pair in pairs:
            if len(pair['ground_truth_indices']) != expected_gt_per_query:
                print(f"❌ Query {pair['query_idx']} in {split_name} has {len(pair['ground_truth_indices'])} ground truth vectors, expected {expected_gt_per_query}")
                print(pair['ground_truth_indices'])
                validation_passed = False
                break
        else:
            continue
        break
    else:
        print(f"✓ Each query has {expected_gt_per_query} ground truth vectors")
    
    if use_mlp_transformations:
        pass
    else:
        # Verify rotation matrices are orthogonal
        for i, rot_matrix in enumerate(transformation_matrices):
            # Check if matrix is orthogonal: R @ R^T = I
            identity_check = rot_matrix @ rot_matrix.T
            if not np.allclose(identity_check, np.eye(config['dimensions']), atol=1e-10):
                print(f"❌ Rotation matrix {i} is not orthogonal")
                validation_passed = False
                break
            
            # Check determinant is 1 (proper rotation, not reflection)
            det = np.linalg.det(rot_matrix)
            if not np.isclose(det, 1.0, atol=1e-10):
                print(f"❌ Rotation matrix {i} determinant is {det}, expected 1.0")
                validation_passed = False
                break
        else:
            print(f"✓ All rotation matrices are proper orthogonal matrices")
    
    if validation_passed:
        print("\n✅ All data integrity checks passed!")
    else:
        print("\n❌ Some data integrity checks failed!")
    
    return validation_passed


def compute_euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(v1 - v2)


def compute_cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    # Cosine distance = 1 - cosine similarity
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norms == 0:
        return 1.0  # Perpendicular vectors
    cosine_sim = dot_product / norms
    return cosine_sim
    # return 1.0 - cosine_sim


def compute_distance_statistics(data: Dict, n_samples: int = 1000, distance_metric: str = 'euclidean') -> Dict:
    """
    Compute various distance statistics on the synthetic data.
    
    Args:
        data: Dictionary containing all loaded data
        n_samples: Number of random samples to use for statistics
        distance_metric: 'euclidean' or 'cosine'
        
    Returns:
        Dictionary containing computed statistics
    """
    print(f"\n=== Computing Distance Statistics ({distance_metric.title()} Distance) ===")
    print(f"Using {n_samples} random samples for each statistic...")
    
    corpus = data['corpus']
    queries = data['queries']
    pairs_data = data['pairs_data']
    config = data['config']
    
    # Choose distance function
    if distance_metric == 'euclidean':
        distance_fn = compute_euclidean_distance
    elif distance_metric == 'cosine':
        distance_fn = compute_cosine_distance
    else:
        raise ValueError(f"Unknown distance metric: {distance_metric}")
    
    # Collect all ground truth indices for each query
    query_to_gt_indices = {}
    all_pairs = pairs_data['train'] + pairs_data['test']
    for pair in all_pairs:
        query_to_gt_indices[pair['query_idx']] = pair['ground_truth_indices']
    
    statistics = {}
    
    # 1. Average distance between two random ground truth vectors belonging to the same query
    print("Computing: Distance between ground truth vectors from same query...")
    same_query_distances = []
    query_indices = list(query_to_gt_indices.keys())
    
    for _ in range(n_samples):
        # Pick a random query
        query_idx = random.choice(query_indices)
        gt_indices = query_to_gt_indices[query_idx]
        
        if len(gt_indices) >= 2:
            # Pick two different ground truth vectors for this query
            idx1, idx2 = random.sample(gt_indices, 2)
            distance = distance_fn(corpus[idx1], corpus[idx2])
            same_query_distances.append(distance)
    
    statistics['avg_distance_same_query_gt'] = np.mean(same_query_distances)
    statistics['std_distance_same_query_gt'] = np.std(same_query_distances)
    print(f"  Mean: {statistics['avg_distance_same_query_gt']:.4f}")
    print(f"  Std:  {statistics['std_distance_same_query_gt']:.4f}")
    
    # Average distance between the second and the third ground truth vector from the same query
    print("Computing: Distance between second and third ground truth vectors from same query...")
    second_third_query_distances = []
    
    for _ in range(n_samples):
        # pick out the second and the third ground truth vector from the same query
        query_idx = random.choice(query_indices)
        gt_indices = query_to_gt_indices[query_idx]
        idx1, idx2 = gt_indices[1], gt_indices[2]
        distance = distance_fn(corpus[idx1], corpus[idx2])
        second_third_query_distances.append(distance)
    
    statistics['avg_distance_second_third_query_gt'] = np.mean(second_third_query_distances)
    statistics['std_distance_second_third_query_gt'] = np.std(second_third_query_distances)
    print(f"  Mean: {statistics['avg_distance_second_third_query_gt']:.4f}")
    print(f"  Std:  {statistics['std_distance_second_third_query_gt']:.4f}")
    
    
    
    
    # 2. Average distance between two random ground truth vectors from different queries
    print("Computing: Distance between ground truth vectors from different queries...")
    diff_query_distances = []
    
    for _ in range(n_samples):
        # Pick two different queries
        query1_idx, query2_idx = random.sample(query_indices, 2)
        gt_indices1 = query_to_gt_indices[query1_idx]
        gt_indices2 = query_to_gt_indices[query2_idx]
        
        # Pick one ground truth vector from each query
        idx1 = random.choice(gt_indices1)
        idx2 = random.choice(gt_indices2)
        distance = distance_fn(corpus[idx1], corpus[idx2])
        diff_query_distances.append(distance)
    
    statistics['avg_distance_diff_query_gt'] = np.mean(diff_query_distances)
    statistics['std_distance_diff_query_gt'] = np.std(diff_query_distances)
    print(f"  Mean: {statistics['avg_distance_diff_query_gt']:.4f}")
    print(f"  Std:  {statistics['std_distance_diff_query_gt']:.4f}")
    
    # 3. Average distance between the query and a random ground truth vector
    print("Computing: Distance between queries and their ground truth vectors...")
    query_gt_distances = []
    
    for _ in range(n_samples):
        # Pick a random query
        query_idx = random.choice(query_indices)
        gt_indices = query_to_gt_indices[query_idx]
        
        # Pick a random ground truth vector for this query
        gt_idx = random.choice(gt_indices)
        distance = distance_fn(queries[query_idx], corpus[gt_idx])
        query_gt_distances.append(distance)
    
    statistics['avg_distance_query_to_gt'] = np.mean(query_gt_distances)
    statistics['std_distance_query_to_gt'] = np.std(query_gt_distances)
    print(f"  Mean: {statistics['avg_distance_query_to_gt']:.4f}")
    print(f"  Std:  {statistics['std_distance_query_to_gt']:.4f}")
    
    # 4. Average distance between two queries
    print("Computing: Distance between different queries...")
    query_query_distances = []
    
    for _ in range(n_samples):
        # Pick two different queries
        query1_idx, query2_idx = random.sample(query_indices, 2)
        distance = distance_fn(queries[query1_idx], queries[query2_idx])
        query_query_distances.append(distance)
    
    statistics['avg_distance_between_queries'] = np.mean(query_query_distances)
    statistics['std_distance_between_queries'] = np.std(query_query_distances)
    print(f"  Mean: {statistics['avg_distance_between_queries']:.4f}")
    print(f"  Std:  {statistics['std_distance_between_queries']:.4f}")
    
    # Additional statistics: distance to random corpus vectors
    print("Computing: Distance between queries and random corpus vectors...")
    query_random_distances = []
    
    for _ in range(n_samples):
        # Pick a random query and random corpus vector
        query_idx = random.choice(query_indices)
        corpus_idx = random.randint(0, len(corpus) - 1)
        distance = distance_fn(queries[query_idx], corpus[corpus_idx])
        query_random_distances.append(distance)
    
    statistics['avg_distance_query_to_random'] = np.mean(query_random_distances)
    statistics['std_distance_query_to_random'] = np.std(query_random_distances)
    print(f"  Mean: {statistics['avg_distance_query_to_random']:.4f}")
    print(f"  Std:  {statistics['std_distance_query_to_random']:.4f}")
    
    return statistics


def print_summary_statistics(stats_euclidean: Dict, stats_cosine: Dict):
    """Print a summary comparison of the statistics."""
    print("\n=== Summary Statistics ===")
    print(f"{'Metric':<40} {'Euclidean':<12} {'Cosine':<12}")
    print("-" * 64)
    
    metrics = [
        ('Same query GT vectors', 'avg_distance_same_query_gt'),
        ('Different query GT vectors', 'avg_distance_diff_query_gt'),
        ('Query to GT vector', 'avg_distance_query_to_gt'),
        ('Between queries', 'avg_distance_between_queries'),
        ('Query to random corpus', 'avg_distance_query_to_random')
    ]
    
    for name, key in metrics:
        euclidean_val = stats_euclidean[key]
        cosine_val = stats_cosine[key]
        print(f"{name:<40} {euclidean_val:<12.4f} {cosine_val:<12.4f}")


def main():
    parser = argparse.ArgumentParser(description='Validate synthetic information retrieval data')
    parser.add_argument('--data-dir', '-d', type=str, default='./synthetic_data',
                       help='Directory containing synthetic data (default: ./synthetic_data)')
    parser.add_argument('--samples', '-n', type=int, default=1000,
                       help='Number of samples for distance statistics (default: 1000)')
    parser.add_argument('--skip-integrity', action='store_true',
                       help='Skip data integrity validation')
    parser.add_argument('--use-mlp-transformations', action='store_true',
                       help='Use MLP transformations')
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Data directory '{args.data_dir}' not found!")
        print("Please run 'python generate_data.py' first to generate the data.")
        return
    
    try:
        # Load the data
        data = load_synthetic_data(args.data_dir)
        
        # Validate data integrity
        if not args.skip_integrity:
            integrity_passed = validate_data_integrity(data, use_mlp_transformations=args.use_mlp_transformations)
            if not integrity_passed:
                print("❌ Data integrity validation failed. Please check the data generation.")
                return
        
        # Set random seed for reproducible statistics
        random.seed(42)
        np.random.seed(42)
        
        # Compute distance statistics
        stats_euclidean = compute_distance_statistics(data, args.samples, 'euclidean')
        stats_cosine = compute_distance_statistics(data, args.samples, 'cosine')
        
        # Print summary
        print_summary_statistics(stats_euclidean, stats_cosine)
        
        print(f"\n✅ Data validation completed successfully!")
        print(f"Dataset contains:")
        print(f"  - {data['config']['n_train_queries']} training queries")
        print(f"  - {data['config']['n_test_queries']} test queries") 
        print(f"  - {data['config']['corpus_size']} corpus vectors")
        print(f"  - {data['config']['dimensions']} dimensions")
        print(f"  - {data['config']['n_transformations']} rotation matrices")
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 