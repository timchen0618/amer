#!/usr/bin/env python3
"""
Opposing Pairs Synthetic Data Generator for Information Retrieval Evaluation

This generator creates a synthetic dataset using opposing pairs transformations:
- 1 Identity matrix (I)
- 2 Linear transformations (A, B) 
- 2 Negative transformations (-A, -B)

This creates an anti-averaging effect that makes single-vector retrieval more challenging.
"""

import numpy as np
import json
import os
from typing import Tuple, List, Dict, Any
from scipy.linalg import qr
import argparse


def convert_to_serializable(obj):
    """
    Convert NumPy data types to native Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj


class OpposingPairsSyntheticDataGenerator:
    """
    Generator for synthetic information retrieval dataset with Gaussian queries
    and opposing pairs ground truth vectors.
    
    Uses 5 transformation matrices:
    - 1 Identity matrix (I)
    - 2 Linear transformations (A, B) 
    - 2 Negative transformations (-A, -B)
    
    This creates an opposing pairs structure that makes single-vector averaging less effective.
    """
    
    def __init__(self, 
                 d: int = 128,
                 n_train_queries: int = 2000,
                 n_test_queries: int = 200,
                 n_transformations: int = 5,  # Fixed to 5 for opposing pairs
                 corpus_size: int = 100000,
                 random_seed: int = 42):
        """
        Initialize the opposing pairs synthetic data generator.
        
        Args:
            d: Dimensionality of vectors
            n_train_queries: Number of training queries
            n_test_queries: Number of test queries  
            n_transformations: Number of transformation matrices (fixed to 5)
            corpus_size: Total size of the corpus
            random_seed: Random seed for reproducibility
        """
        self.d = d
        self.n_train_queries = n_train_queries
        self.n_test_queries = n_test_queries
        self.n_total_queries = n_train_queries + n_test_queries
        self.n_transformations = 5  # Always 5 for opposing pairs approach
        self.corpus_size = corpus_size
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Will store generated data
        self.queries = None
        self.transformation_matrices = None
        self.ground_truth_vectors = None
        self.corpus = None
        self.query2ground_truth_mapping = None
        
    def generate_queries(self) -> np.ndarray:
        """
        Generate query vectors from multivariate Gaussian N(0, I).
        
        Returns:
            Array of shape (n_total_queries, d)
        """
        print(f"Generating {self.n_total_queries} query vectors of dimension {self.d}...")
        queries = np.random.multivariate_normal(
            mean=np.zeros(self.d),
            cov=np.eye(self.d),
            size=self.n_total_queries
        )
        self.queries = queries
        return queries
    
    def generate_transformation_matrices(self) -> List[np.ndarray]:
        """
        Generate transformation matrices: 1 identity + 2 linear transformations + 2 negatives.
        This creates an opposing pairs structure: [I, A, B, -A, -B]
        
        Returns:
            List of 5 transformation matrices, each of shape (d, d)
        """
        print(f"Generating opposing pairs transformation matrices...")
        
        transformation_matrices = []
        
        # 1. Identity matrix
        identity = np.eye(self.d)
        transformation_matrices.append(identity)
        print("  ✓ Added identity matrix (I)")
        
        # 2. Generate 2 linear transformations using QR decomposition
        linear_transforms = []
        for i in range(2):
            # Generate random matrix and get orthogonal matrix via QR decomposition
            random_matrix = np.random.randn(self.d, self.d)
            q, r = qr(random_matrix)
            
            # Ensure determinant is 1 (proper rotation, not reflection)
            if np.linalg.det(q) < 0:
                q[:, 0] *= -1
            
            linear_transforms.append(q)
            transformation_matrices.append(q)
            print(f"  ✓ Added linear transformation {chr(65+i)} (matrix {i+2})")
        
        # 3. Add negative versions of the linear transformations
        for i, transform in enumerate(linear_transforms):
            neg_transform = -transform
            transformation_matrices.append(neg_transform)
            print(f"  ✓ Added negative transformation -{chr(65+i)} (matrix {i+4})")
        
        self.transformation_matrices = transformation_matrices
        print(f"Generated {len(transformation_matrices)} transformation matrices")
        self._print_matrix_relationships()
        self._test_opposing_pairs_effect()
        return transformation_matrices
    
    def _print_matrix_relationships(self):
        """Print relationships between transformation matrices."""
        print("\nTransformation matrix structure:")
        print("  Matrix 1: Identity (I)")
        print("  Matrix 2: Linear transformation A")
        print("  Matrix 3: Linear transformation B") 
        print("  Matrix 4: Negative transformation -A")
        print("  Matrix 5: Negative transformation -B")
        print("  → Opposing pairs: (A, -A) and (B, -B) cancel when averaged")
    
    def _test_opposing_pairs_effect(self):
        """Test the anti-averaging effect of opposing pairs."""
        print("\nTesting opposing pairs effect:")
        
        # Generate a sample query
        sample_query = np.random.randn(self.d)
        sample_query = sample_query / np.linalg.norm(sample_query)
        
        # Apply all transformations
        transformed_vectors = []
        for i, matrix in enumerate(self.transformation_matrices):
            transformed = matrix @ sample_query
            transformed_vectors.append(transformed)
        
        # Compute average
        average_vector = np.mean(transformed_vectors, axis=0)
        
        # Compute similarities
        similarities = []
        for i, vec in enumerate(transformed_vectors):
            sim = np.dot(average_vector, vec)
            similarities.append(sim)
            print(f"  Similarity(avg, transform_{i+1}): {sim:.3f}")
        
        avg_similarity = np.mean(similarities)
        print(f"  Average similarity: {avg_similarity:.3f}")
        print(f"  Expected difficulty: {'HARD' if avg_similarity < 0.5 else 'MEDIUM' if avg_similarity < 0.7 else 'EASY'}")
    
    def generate_ground_truth_vectors(self) -> np.ndarray:
        """
        Generate ground truth vectors by applying transformation matrices to queries.
        For each query x, creates y_i = T_i @ x for i=1,...,5 where T = [I, A, B, -A, -B]
        
        Returns:
            Array of shape (n_total_queries * n_transformations, d)
        """
        if self.queries is None or self.transformation_matrices is None:
            raise ValueError("Must generate queries and transformation matrices first")
            
        print(f"Generating ground truth vectors using opposing pairs...")
        
        ground_truth_vectors = []
        
        # Generate ground truth vectors organized by query:
        # [Q0T0, Q0T1, Q0T2, Q0T3, Q0T4, Q1T0, Q1T1, Q1T2, Q1T3, Q1T4, ...]
        for query_idx, query in enumerate(self.queries):
            for transform_idx, transformation_matrix in enumerate(self.transformation_matrices):
                # Apply transformation to this specific query
                transformed_query = query @ transformation_matrix.T
                ground_truth_vectors.append(transformed_query)
        
        # Convert to numpy array
        self.ground_truth_vectors = np.array(ground_truth_vectors)
        print(f"Generated {self.ground_truth_vectors.shape[0]} ground truth vectors")
        print(f"Organization: [Q0T0, Q0T1, ..., Q0T4, Q1T0, Q1T1, ..., Q1T4, ...]")
        return self.ground_truth_vectors
    
    def generate_corpus(self) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """
        Generate the full corpus by combining ground truth vectors with random vectors.
        
        Returns:
            Tuple of (corpus, query2ground_truth_mapping)
        """
        if self.ground_truth_vectors is None:
            raise ValueError("Must generate ground truth vectors first")
            
        n_ground_truth = self.ground_truth_vectors.shape[0]
        n_random = self.corpus_size - n_ground_truth
        
        if n_random < 0:
            raise ValueError(f"Corpus size {self.corpus_size} is smaller than ground truth vectors {n_ground_truth}")
        
        print(f"Generating corpus with {n_ground_truth} ground truth vectors and {n_random} random vectors...")
        
        # Generate random vectors to fill the corpus
        random_vectors = np.random.randn(n_random, self.d)
        
        # Combine ground truth and random vectors
        corpus = np.vstack([self.ground_truth_vectors, random_vectors])

        # Create mapping from query index to ground truth index in the corpus
        query2ground_truth_mapping = {}
        for i in range(self.n_total_queries):
            query2ground_truth_mapping[i] = [i*self.n_transformations + j for j in range(self.n_transformations)]
        
        self.corpus = corpus
        self.query2ground_truth_mapping = query2ground_truth_mapping
        
        print(f"Generated corpus of size {corpus.shape}")
        return corpus, query2ground_truth_mapping
    
    def create_query_ground_truth_pairs(self) -> Dict[str, Any]:
        """
        Create query-ground truth pairs for training and testing.
        
        Returns:
            Dictionary containing train/test splits with query-ground truth mappings
        """
        if self.queries is None or self.query2ground_truth_mapping is None:
            raise ValueError("Must generate all data first")
        
        print("Creating query-ground truth pairs...")
        
        # Split queries into train/test
        train_queries = self.queries[:self.n_train_queries]
        test_queries = self.queries[self.n_train_queries:]
        
        # Create mappings for each split
        train_pairs = []
        test_pairs = []
        
        # For each query, find its ground truth vectors in the corpus
        for query_idx in range(self.n_total_queries):
            is_train = query_idx < self.n_train_queries
            
            pair = {
                'query_idx': int(query_idx),  # Convert to native int
                'query_vector': self.queries[query_idx].tolist(),  # Convert to list
                'ground_truth_indices': self.query2ground_truth_mapping[query_idx]
            }
            
            if is_train:
                train_pairs.append(pair)
            else:
                test_pairs.append(pair)
        
        pairs_data = {
            'train': train_pairs,
            'test': test_pairs,
            'metadata': {
                'n_train_queries': int(self.n_train_queries),
                'n_test_queries': int(self.n_test_queries),
                'n_transformations': int(self.n_transformations),
                'dimensions': int(self.d),
                'corpus_size': int(self.corpus_size),
                'transformation_type': 'opposing_pairs'
            }
        }
        
        print(f"Created {len(train_pairs)} training pairs and {len(test_pairs)} test pairs")
        return pairs_data
    
    def save_data(self, output_dir: str):
        """
        Save all generated data to files.
        
        Args:
            output_dir: Directory to save data files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving data to {output_dir}...")
        
        # Save corpus
        np.save(os.path.join(output_dir, 'corpus.npy'), self.corpus)
        
        # Save transformation matrices
        transformation_matrices_array = np.array(self.transformation_matrices)
        np.save(os.path.join(output_dir, 'transformation_matrices.npy'), transformation_matrices_array)
        
        # Save queries
        np.save(os.path.join(output_dir, 'queries.npy'), self.queries)
        
        # Save query-ground truth pairs
        pairs_data = self.create_query_ground_truth_pairs()
        # Convert all data to JSON-serializable format
        pairs_data_serializable = convert_to_serializable(pairs_data)
        with open(os.path.join(output_dir, 'query_ground_truth_pairs.json'), 'w') as f:
            json.dump(pairs_data_serializable, f, indent=4)
        
        # Save configuration
        config = {
            'dimensions': int(self.d),
            'n_train_queries': int(self.n_train_queries),
            'n_test_queries': int(self.n_test_queries),
            'n_transformations': int(self.n_transformations),
            'corpus_size': int(self.corpus_size),
            'random_seed': int(self.random_seed),
            'transformation_type': 'opposing_pairs',
            'transformations': 'identity + 2_linear + 2_negative',
            'approach': 'I + A + B + (-A) + (-B)'
        }
        # Convert config to JSON-serializable format
        config_serializable = convert_to_serializable(config)
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config_serializable, f, indent=4)
        
        print("Data saved successfully!")
        print(f"Files created:")
        print(f"  - corpus.npy: {self.corpus.shape}")
        print(f"  - queries.npy: {self.queries.shape}")
        print(f"  - transformation_matrices.npy: {transformation_matrices_array.shape}")
        print(f"  - query_ground_truth_pairs.json")
        print(f"  - config.json")
    
    def generate_all(self) -> Dict[str, Any]:
        """
        Generate all components of the opposing pairs synthetic dataset.
        
        Returns:
            Dictionary containing all generated data
        """
        print("=== Starting Opposing Pairs Synthetic Data Generation ===")
        print(f"Configuration:")
        print(f"  Dimensions: {self.d}")
        print(f"  Training queries: {self.n_train_queries}")
        print(f"  Test queries: {self.n_test_queries}")
        print(f"  Transformations: {self.n_transformations} (opposing pairs)")
        print(f"  Corpus size: {self.corpus_size}")
        print(f"  Random seed: {self.random_seed}")
        print(f"  Approach: I + A + B + (-A) + (-B)")
        print()
        
        # Generate all components
        self.generate_queries()
        self.generate_transformation_matrices()
        self.generate_ground_truth_vectors()
        self.generate_corpus()
        
        print("\n=== Generation Complete ===")
        print("This dataset uses opposing pairs to create anti-averaging effects!")
        
        return {
            'queries': self.queries,
            'transformation_matrices': self.transformation_matrices,
            'ground_truth_vectors': self.ground_truth_vectors,
            'corpus': self.corpus,
            'query2ground_truth_mapping': self.query2ground_truth_mapping
        }


def main():
    parser = argparse.ArgumentParser(description='Generate opposing pairs synthetic data for information retrieval evaluation')
    parser.add_argument('--dimensions', '-d', type=int, default=128, 
                       help='Dimensionality of vectors (default: 128)')
    parser.add_argument('--train-queries', type=int, default=2000,
                       help='Number of training queries (default: 2000)')
    parser.add_argument('--test-queries', type=int, default=200,
                       help='Number of test queries (default: 200)')
    parser.add_argument('--corpus-size', type=int, default=100000,
                       help='Total corpus size (default: 100000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', '-o', type=str, default='./opposing_pairs_data',
                       help='Output directory (default: ./opposing_pairs_data)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = OpposingPairsSyntheticDataGenerator(
        d=args.dimensions,
        n_train_queries=args.train_queries,
        n_test_queries=args.test_queries,
        corpus_size=args.corpus_size,
        random_seed=args.seed
    )
    
    # Generate all data
    data = generator.generate_all()
    
    # Save data
    generator.save_data(args.output_dir)
    
    print(f"\n🎯 Opposing pairs dataset generated!")
    print(f"   Expected to be harder than original rotation-based approach.")
    print(f"   Run baseline evaluation to compare:")
    print(f"   python baseline_evaluation.py --data-dir {args.output_dir}")


if __name__ == '__main__':
    main() 