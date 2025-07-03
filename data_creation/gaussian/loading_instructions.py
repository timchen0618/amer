#!/usr/bin/env python3
"""
Simple instructions and examples for loading the synthetic dataset.
"""

import numpy as np
import json
import os


def load_synthetic_dataset(data_dir='./synthetic_data'):
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
    corpus = np.load(os.path.join(data_dir, 'corpus.npy'))              # Shape: (corpus_size, dimensions)
    queries = np.load(os.path.join(data_dir, 'queries.npy'))            # Shape: (total_queries, dimensions)
    rotation_matrices = np.load(os.path.join(data_dir, 'rotation_matrices.npy'))  # Shape: (n_rotations, dimensions, dimensions)
    ground_truth_indices = np.load(os.path.join(data_dir, 'ground_truth_indices.npy'))  # Shape: (corpus_size,) - boolean array
    
    # 3. Load query-ground truth mappings
    with open(os.path.join(data_dir, 'query_ground_truth_pairs.json'), 'r') as f:
        pairs_data = json.load(f)
    
    return {
        'config': config,
        'corpus': corpus,
        'queries': queries,
        'rotation_matrices': rotation_matrices,
        'ground_truth_indices': ground_truth_indices,
        'pairs_data': pairs_data
    }


def example_usage():
    """Example of how to use the loaded data."""
    
    print("=== Loading Synthetic Dataset ===")
    
    # Load the dataset
    data = load_synthetic_dataset('./synthetic_data')
    
    print(f"Dataset configuration:")
    print(f"  Dimensions: {data['config']['dimensions']}")
    print(f"  Training queries: {data['config']['n_train_queries']}")
    print(f"  Test queries: {data['config']['n_test_queries']}")
    print(f"  Corpus size: {data['config']['corpus_size']}")
    print(f"  Number of rotations: {data['config']['n_rotations']}")
    
    print(f"\nData shapes:")
    print(f"  Corpus: {data['corpus'].shape}")
    print(f"  Queries: {data['queries'].shape}")
    print(f"  Rotation matrices: {data['rotation_matrices'].shape}")
    
    # Example 1: Access training and test data
    print(f"\n=== Train/Test Split ===")
    train_pairs = data['pairs_data']['train']
    test_pairs = data['pairs_data']['test']
    
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Test pairs: {len(test_pairs)}")
    
    # Example 2: Get a specific query and its ground truth
    print(f"\n=== Example Query ===")
    example_query = train_pairs[0]
    query_idx = example_query['query_idx']
    query_vector = np.array(example_query['query_vector'])
    gt_indices = example_query['ground_truth_indices']
    
    print(f"Query {query_idx}:")
    print(f"  Query vector shape: {query_vector.shape}")
    print(f"  Ground truth indices in corpus: {gt_indices}")
    
    # Retrieve ground truth vectors from corpus
    gt_vectors = data['corpus'][gt_indices]
    print(f"  Ground truth vectors shape: {gt_vectors.shape}")
    
    # Example 3: Compute similarities
    print(f"\n=== Computing Similarities ===")
    # Cosine similarity between query and its ground truth vectors
    query_norm = query_vector / np.linalg.norm(query_vector)
    gt_norms = gt_vectors / np.linalg.norm(gt_vectors, axis=1, keepdims=True)
    similarities = gt_norms @ query_norm
    
    print(f"Cosine similarities with ground truth vectors: {similarities}")
    
    # Example 4: Extract training data for a model
    print(f"\n=== Preparing Training Data ===")
    
    # Get all training queries
    train_query_indices = [pair['query_idx'] for pair in train_pairs]
    train_queries = data['queries'][train_query_indices]
    
    print(f"Training queries shape: {train_queries.shape}")
    
    # Create a mapping from query index to ground truth indices
    query_to_gt = {}
    for pair in train_pairs:
        query_to_gt[pair['query_idx']] = pair['ground_truth_indices']
    
    print(f"Query-to-ground-truth mapping created for {len(query_to_gt)} queries")
    
    # Example 5: Access corpus vectors by type
    print(f"\n=== Corpus Analysis ===")
    ground_truth_mask = data['ground_truth_indices']
    n_ground_truth = np.sum(ground_truth_mask)
    n_random = len(ground_truth_mask) - n_ground_truth
    
    print(f"Corpus composition:")
    print(f"  Ground truth vectors: {n_ground_truth}")
    print(f"  Random vectors: {n_random}")
    print(f"  Total: {len(data['corpus'])}")
    
    # Get only the ground truth vectors
    gt_vectors_all = data['corpus'][ground_truth_mask]
    random_vectors = data['corpus'][~ground_truth_mask]
    
    print(f"  Ground truth vectors shape: {gt_vectors_all.shape}")
    print(f"  Random vectors shape: {random_vectors.shape}")


if __name__ == '__main__':
    # Instructions
    print("""
# How to Load Synthetic Dataset

## Quick Start

```python
import numpy as np
import json

# Load the dataset
data = load_synthetic_dataset('./synthetic_data')

# Access components
corpus = data['corpus']                    # All searchable vectors
queries = data['queries']                  # Query vectors  
train_pairs = data['pairs_data']['train']  # Training query-GT pairs
test_pairs = data['pairs_data']['test']    # Test query-GT pairs
config = data['config']                    # Dataset metadata
```

## File Structure

- `corpus.npy`: All vectors that can be searched (includes GT + random)
- `queries.npy`: All query vectors (train + test)
- `rotation_matrices.npy`: The rotation matrices used for GT generation
- `ground_truth_indices.npy`: Boolean mask indicating GT vectors in corpus
- `query_ground_truth_pairs.json`: Query-GT mappings with train/test split
- `config.json`: Dataset configuration parameters

## Key Data Access Patterns

1. **Get training data**: `train_pairs = data['pairs_data']['train']`
2. **Get test data**: `test_pairs = data['pairs_data']['test']`
3. **Get ground truth for query**: `gt_indices = pair['ground_truth_indices']`
4. **Get GT vectors**: `gt_vectors = corpus[gt_indices]`
5. **Filter corpus**: `gt_only = corpus[ground_truth_indices]`

## Running the Example
""")
    
    # Check if data exists
    if os.path.exists('./synthetic_data'):
        example_usage()
    else:
        print("❌ No data found at './synthetic_data'")
        print("Run 'python generate_data.py' first to generate the dataset.") 