#!/usr/bin/env python3
"""
Debug script to understand ground truth vector organization.
"""

import numpy as np
import json
import os


def debug_ground_truth_structure(data_dir):
    """Debug the ground truth vector structure."""
    print("=== DEBUGGING GROUND TRUTH STRUCTURE ===")
    
    # Load data
    corpus = np.load(os.path.join(data_dir, 'corpus.npy'))
    queries = np.load(os.path.join(data_dir, 'queries.npy'))
    
    with open(os.path.join(data_dir, 'query_ground_truth_pairs.json'), 'r') as f:
        pairs_data = json.load(f)
    
    # Check a few test queries
    test_pairs = pairs_data['test'][:3]  # First 3 test queries
    
    for i, pair in enumerate(test_pairs):
        query_idx = pair['query_idx']
        gt_indices = pair['ground_truth_indices']
        
        print(f"\n--- Test Query {i} (Global Index {query_idx}) ---")
        print(f"Ground truth indices in corpus: {gt_indices}")
        
        # Get the query vector
        query_vec = queries[query_idx]
        print(f"Query vector norm: {np.linalg.norm(query_vec):.4f}")
        
        # Get the ground truth vectors
        gt_vectors = corpus[gt_indices]
        print(f"Ground truth vectors shape: {gt_vectors.shape}")
        
        # Check similarity between query and each ground truth
        similarities = []
        for j, gt_vec in enumerate(gt_vectors):
            similarity = np.dot(query_vec, gt_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(gt_vec))
            similarities.append(similarity)
            print(f"  GT {j}: similarity = {similarity:.4f}, norm = {np.linalg.norm(gt_vec):.4f}")
        
        # Check if identity transformation exists
        max_sim = max(similarities)
        if max_sim > 0.98:
            print(f"  ✅ Found identity-like transformation (similarity = {max_sim:.4f})")
        else:
            print(f"  ❌ No identity-like transformation found (max similarity = {max_sim:.4f})")
        
        # Check average of ground truth vectors
        avg_gt = np.mean(gt_vectors, axis=0)
        avg_similarity = np.dot(query_vec, avg_gt) / (np.linalg.norm(query_vec) * np.linalg.norm(avg_gt))
        print(f"  Average GT similarity: {avg_similarity:.4f}")


def debug_transformation_matrices(data_dir):
    """Debug the transformation matrices structure."""
    print("\n=== DEBUGGING TRANSFORMATION MATRICES ===")
    
    try:
        # Try to load transformation matrices
        transform_matrices = np.load(os.path.join(data_dir, 'transformation_matrices.npy'))
        print(f"Transformation matrices shape: {transform_matrices.shape}")
        
        # Check if first matrix is identity
        identity_check = np.allclose(transform_matrices[0], np.eye(transform_matrices[0].shape[0]))
        print(f"First matrix is identity: {identity_check}")
        
        if identity_check:
            print("  ✅ Identity matrix found at index 0")
        else:
            print("  ❌ First matrix is not identity")
            print(f"  First matrix diagonal: {np.diag(transform_matrices[0])[:5]}...")
            
    except FileNotFoundError:
        print("No transformation_matrices.npy file found")


def test_manual_mapping(data_dir):
    """Test manual computation of ground truth mapping."""
    print("\n=== TESTING MANUAL MAPPING ===")
    
    # Load data
    queries = np.load(os.path.join(data_dir, 'queries.npy'))
    
    try:
        transform_matrices = np.load(os.path.join(data_dir, 'transformation_matrices.npy'))
        
        # Manually compute ground truth for first test query
        test_query_idx = 2000  # First test query (assuming 2000 train queries)
        if test_query_idx >= len(queries):
            test_query_idx = len(queries) - 1
            
        query_vec = queries[test_query_idx]
        print(f"Test query index: {test_query_idx}")
        print(f"Query vector norm: {np.linalg.norm(query_vec):.4f}")
        
        # Apply all transformations manually
        manual_gt_vectors = []
        for i, transform in enumerate(transform_matrices):
            gt_vec = query_vec @ transform.T
            manual_gt_vectors.append(gt_vec)
            similarity = np.dot(query_vec, gt_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(gt_vec))
            print(f"  Transform {i}: similarity = {similarity:.4f}")
            
        # Check where these should be in the corpus
        # According to the current organization: transformation i, all queries
        # So for query j and transformation i: index = i * n_total_queries + j
        n_total_queries = len(queries)
        expected_indices = []
        for i in range(len(transform_matrices)):
            expected_idx = i * n_total_queries + test_query_idx
            expected_indices.append(expected_idx)
        
        print(f"Expected corpus indices: {expected_indices}")
        
    except FileNotFoundError:
        print("No transformation matrices found - this might be using a different generator")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python debug_ground_truth_structure.py <data_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    debug_ground_truth_structure(data_dir)
    debug_transformation_matrices(data_dir)
    test_manual_mapping(data_dir) 