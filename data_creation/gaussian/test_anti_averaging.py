#!/usr/bin/env python3
"""
Diagnostic script to test anti-averaging effectiveness.
"""

import numpy as np
from generate_data_hard import HardSyntheticDataGenerator


def test_anti_averaging_effect():
    """Test how well the anti-averaging strategy works."""
    print("=== Testing Anti-Averaging Strategy ===")
    
    # Create generator
    generator = HardSyntheticDataGenerator(d=64, n_train_queries=5, n_test_queries=0, corpus_size=100)
    
    # Generate a few queries and test
    generator.generate_queries()
    generator.generate_ground_truth_vectors()
    
    print(f"\nAnalyzing first 3 queries:")
    
    for q_idx in range(3):
        query = generator.queries[q_idx]
        
        # Get ground truth vectors for this query
        start_idx = q_idx * 5
        end_idx = start_idx + 5
        gt_vecs = generator.ground_truth_vectors[start_idx:end_idx]
        
        # Compute average
        avg_vec = np.mean(gt_vecs, axis=0)
        avg_vec = avg_vec / np.linalg.norm(avg_vec)
        
        # Compute similarities
        query_to_avg = np.dot(query, avg_vec)
        avg_to_gt_sims = [np.dot(avg_vec, gt) for gt in gt_vecs]
        query_to_gt_sims = [np.dot(query, gt) for gt in gt_vecs]
        
        print(f"\nQuery {q_idx}:")
        print(f"  Query -> Average GT similarity: {query_to_avg:.3f}")
        print(f"  Average GT -> Individual GT similarities: {[f'{s:.3f}' for s in avg_to_gt_sims]}")
        print(f"  Query -> Individual GT similarities: {[f'{s:.3f}' for s in query_to_gt_sims]}")
        print(f"  Average of (Avg->GT): {np.mean(avg_to_gt_sims):.3f}")
        print(f"  Min (Avg->GT): {np.min(avg_to_gt_sims):.3f}")
        
        # Check if vectors are actually spread out
        gt_pairwise_sims = []
        for i in range(len(gt_vecs)):
            for j in range(i+1, len(gt_vecs)):
                sim = np.dot(gt_vecs[i], gt_vecs[j])
                gt_pairwise_sims.append(sim)
        print(f"  Pairwise GT similarities: avg={np.mean(gt_pairwise_sims):.3f}, min={np.min(gt_pairwise_sims):.3f}")


def test_extreme_anti_averaging():
    """Test a more extreme anti-averaging approach."""
    print("\n\n=== Testing EXTREME Anti-Averaging ===")
    
    d = 64
    query = np.random.randn(d)
    query = query / np.linalg.norm(query)
    
    # Method 1: Orthogonal vectors
    print("\nMethod 1: Orthogonal vectors")
    gt_vecs_ortho = []
    for i in range(5):
        vec = np.random.randn(d)
        # Make orthogonal to query
        vec = vec - np.dot(vec, query) * query
        vec = vec / np.linalg.norm(vec)
        gt_vecs_ortho.append(vec)
    
    avg_ortho = np.mean(gt_vecs_ortho, axis=0)
    if np.linalg.norm(avg_ortho) > 0:
        avg_ortho = avg_ortho / np.linalg.norm(avg_ortho)
        ortho_sims = [np.dot(avg_ortho, gt) for gt in gt_vecs_ortho]
        print(f"  Average -> GT similarities: {[f'{s:.3f}' for s in ortho_sims]}")
        print(f"  Average similarity: {np.mean(ortho_sims):.3f}")
    else:
        print("  Average vector is zero! Perfect anti-averaging!")
    
    # Method 2: Opposing pairs + one orthogonal
    print("\nMethod 2: Opposing pairs + orthogonal")
    gt_vecs_opposing = []
    
    # Create two opposing pairs
    for i in range(2):
        vec = np.random.randn(d)
        vec = vec / np.linalg.norm(vec)
        gt_vecs_opposing.append(vec)
        gt_vecs_opposing.append(-vec)  # Opposite direction
    
    # Add one orthogonal vector
    ortho_vec = np.random.randn(d)
    for vec in gt_vecs_opposing:
        ortho_vec = ortho_vec - np.dot(ortho_vec, vec) * vec
    ortho_vec = ortho_vec / np.linalg.norm(ortho_vec)
    gt_vecs_opposing.append(ortho_vec)
    
    avg_opposing = np.mean(gt_vecs_opposing, axis=0)
    if np.linalg.norm(avg_opposing) > 0:
        avg_opposing = avg_opposing / np.linalg.norm(avg_opposing)
        opposing_sims = [np.dot(avg_opposing, gt) for gt in gt_vecs_opposing]
        print(f"  Average -> GT similarities: {[f'{s:.3f}' for s in opposing_sims]}")
        print(f"  Average similarity: {np.mean(opposing_sims):.3f}")
    else:
        print("  Average vector is zero! Perfect anti-averaging!")


if __name__ == "__main__":
    test_anti_averaging_effect()
    test_extreme_anti_averaging() 