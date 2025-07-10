#!/usr/bin/env python3
"""
Hard Synthetic Data Generator for Information Retrieval Evaluation

This generator creates a challenging synthetic dataset that's difficult for single-vector retrieval.
It uses:
- Strategy 3: Hierarchical ground truth structure that can't be captured by a single vector
- Strategy 2: Hard negative vectors that are similar to ground truth but not relevant

The resulting dataset forces retrieval systems to go beyond simple averaging approaches.
"""

import numpy as np
import json
import os
from typing import Tuple, List, Dict, Any
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


class HardSyntheticDataGenerator:
    """
    Generator for challenging synthetic information retrieval dataset.
    Creates opposing pairs ground truth vectors and hard negatives that make
    single-vector retrieval perform very poorly.
    
    Uses Strategy 4-Modified (Opposing pairs) + Strategy 2 (Hard negatives):
    - Ground truth vectors include opposing pairs that cancel out when averaged
    - Hard negative vectors similar to ground truth but not relevant
    """
    
    def __init__(self, 
                 d: int = 128,
                 n_train_queries: int = 2000,
                 n_test_queries: int = 200,
                 n_ground_truth_per_query: int = 5,
                 corpus_size: int = 100000,
                 random_seed: int = 42):
        """
        Initialize the hard synthetic data generator.
        
        Args:
            d: Dimensionality of vectors
            n_train_queries: Number of training queries
            n_test_queries: Number of test queries  
            n_ground_truth_per_query: Number of ground truth vectors per query
            corpus_size: Total size of the corpus
            random_seed: Random seed for reproducibility
        """
        self.d = d
        self.n_train_queries = n_train_queries
        self.n_test_queries = n_test_queries
        self.n_total_queries = n_train_queries + n_test_queries
        self.n_ground_truth_per_query = n_ground_truth_per_query
        self.corpus_size = corpus_size
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Will store generated data
        self.queries = None
        self.ground_truth_vectors = None
        self.hard_negatives = None
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
    
    def generate_ground_truth_vectors(self) -> np.ndarray:
        """
        Generate challenging ground truth vectors using hierarchical structure.
        Each query gets multiple ground truth vectors that can't be well-represented by their average.
        
        Returns:
            Array of shape (n_total_queries * n_ground_truth_per_query, d)
        """
        if self.queries is None:
            raise ValueError("Must generate queries first")
            
        print(f"Generating challenging opposing pairs ground truth vectors...")
        
        ground_truth_vectors = []
        self.hard_negatives = []  # Store hard negatives separately
        
        for query_idx, query in enumerate(self.queries):
            # Generate opposing pairs ground truth (Strategy 4)
            gt_vecs = self._generate_hierarchical_ground_truth(query)
            ground_truth_vectors.extend(gt_vecs)
            
            # Generate hard negatives (Strategy 2)
            hard_negs = self._generate_hard_negatives(query, gt_vecs)
            self.hard_negatives.extend(hard_negs)
        
        self.ground_truth_vectors = np.array(ground_truth_vectors)
        self.hard_negatives = np.array(self.hard_negatives)
        
        print(f"Generated {self.ground_truth_vectors.shape[0]} ground truth vectors")
        print(f"Generated {self.hard_negatives.shape[0]} hard negative vectors")
        
        # Test the difficulty by computing average similarities
        self._test_difficulty()
        
        return self.ground_truth_vectors
    
    def _generate_hierarchical_ground_truth(self, query: np.ndarray) -> List[np.ndarray]:
        """
        Generate opposing pairs ground truth that truly breaks averaging.
        
        Creates 2 opposing pairs + 1 orthogonal vector. The opposing pairs
        cancel each other out when averaged, making single-vector retrieval fail.
        """
        gt_vectors = []
        
        # Method: Opposing pairs + orthogonal (most effective from testing)
        
        # Create two opposing pairs (4 vectors total)
        for i in range(2):
            # Generate a random direction
            vec = np.random.randn(self.d)
            vec = vec / np.linalg.norm(vec)
            
            # Add the vector and its exact opposite
            gt_vectors.append(vec)
            gt_vectors.append(-vec)  # Opposite direction - this cancels out in averaging!
        
        # Add one orthogonal vector (5th vector)
        ortho_vec = np.random.randn(self.d)
        
        # Make it orthogonal to all previous vectors
        for vec in gt_vectors:
            ortho_vec = ortho_vec - np.dot(ortho_vec, vec) * vec
        
        # If ortho_vec becomes zero (unlikely), generate a random one
        if np.linalg.norm(ortho_vec) < 1e-6:
            ortho_vec = np.random.randn(self.d)
        
        ortho_vec = ortho_vec / np.linalg.norm(ortho_vec)
        gt_vectors.append(ortho_vec)
        
        return gt_vectors
    
    def _generate_hard_negatives(self, query: np.ndarray, gt_vectors: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate hard negative vectors that are similar to ground truth but not relevant.
        These confound simple similarity-based retrieval.
        """
        hard_negatives = []
        n_hard_negs_per_query = 10  # Increased to 10 hard negatives per query
        
        # Compute average of ground truth vectors as cluster center
        cluster_center = np.mean(gt_vectors, axis=0)
        cluster_center = cluster_center / np.linalg.norm(cluster_center)
        
        # Also create hard negatives similar to individual ground truth vectors
        for i in range(n_hard_negs_per_query):
            if i < 5:
                # First 5: similar to cluster center (averaged GT)
                # Create vector that's 70-85% similar to cluster center
                similarity_target = 0.70 + 0.15 * np.random.random()
                reference_vec = cluster_center
            else:
                # Last 5: similar to individual GT vectors
                gt_idx = (i - 5) % len(gt_vectors)
                # Create vector that's 75-90% similar to individual GT
                similarity_target = 0.75 + 0.15 * np.random.random()
                reference_vec = gt_vectors[gt_idx]
            
            # Generate random direction
            random_dir = np.random.randn(self.d)
            random_dir = random_dir / np.linalg.norm(random_dir)
            
            # Mix with reference vector to achieve target similarity
            hard_neg = similarity_target * reference_vec + np.sqrt(1 - similarity_target**2) * random_dir
            hard_negatives.append(hard_neg / np.linalg.norm(hard_neg))
        
        return hard_negatives
    
    def _test_difficulty(self):
        """Test how difficult this dataset is for single-vector retrieval."""
        if len(self.ground_truth_vectors) == 0:
            return
            
        # Sample a few queries to test difficulty
        n_test_queries = min(5, self.n_total_queries)
        similarities = []
        
        for i in range(n_test_queries):
            start_idx = i * self.n_ground_truth_per_query
            end_idx = start_idx + self.n_ground_truth_per_query
            gt_vecs = self.ground_truth_vectors[start_idx:end_idx]
            
            # Compute average (what single-vector baseline would predict)
            avg_vec = np.mean(gt_vecs, axis=0)
            
            # Compute similarities between average and individual ground truth
            sims = [np.dot(avg_vec, gt) for gt in gt_vecs]
            similarities.extend(sims)
        
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        
        print(f"Difficulty Assessment:")
        print(f"  Average similarity between averaged GT and individual GT: {avg_sim:.3f}")
        print(f"  Minimum similarity: {min_sim:.3f}")
        print(f"  Expected performance: {'HARD' if avg_sim < 0.5 else 'MEDIUM' if avg_sim < 0.7 else 'EASY'}")
    
    def generate_corpus(self) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """
        Generate the full corpus by combining ground truth vectors, hard negatives, and random vectors.
        
        Returns:
            Tuple of (corpus, query2ground_truth_mapping)
        """
        if self.ground_truth_vectors is None or self.hard_negatives is None:
            raise ValueError("Must generate ground truth vectors and hard negatives first")
            
        n_ground_truth = self.ground_truth_vectors.shape[0]
        n_hard_negatives = self.hard_negatives.shape[0]
        n_random = self.corpus_size - n_ground_truth - n_hard_negatives
        
        if n_random < 0:
            raise ValueError(f"Corpus size {self.corpus_size} is smaller than ground truth ({n_ground_truth}) + hard negatives ({n_hard_negatives})")
        
        print(f"Generating corpus with:")
        print(f"  - {n_ground_truth} ground truth vectors")
        print(f"  - {n_hard_negatives} hard negative vectors") 
        print(f"  - {n_random} random vectors")
        
        # Generate random vectors to fill the remaining corpus
        random_vectors = np.random.randn(n_random, self.d)
        
        # Combine all vectors: [ground_truth, hard_negatives, random]
        corpus = np.vstack([
            self.ground_truth_vectors,
            self.hard_negatives,
            random_vectors
        ])

        # Create mapping from query index to ground truth indices in the corpus
        # Ground truth vectors are at the beginning of corpus (indices 0 to n_ground_truth-1)
        query2ground_truth_mapping = {}
        for i in range(self.n_total_queries):
            start_idx = i * self.n_ground_truth_per_query
            end_idx = start_idx + self.n_ground_truth_per_query
            query2ground_truth_mapping[i] = list(range(start_idx, end_idx))
        
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
                'n_ground_truth_per_query': int(self.n_ground_truth_per_query),
                'dimensions': int(self.d),
                'corpus_size': int(self.corpus_size),
                'dataset_type': 'hard_opposing_pairs_with_negatives'
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
        
        # Save queries
        np.save(os.path.join(output_dir, 'queries.npy'), self.queries)
        
        # Save hard negatives (for analysis)
        np.save(os.path.join(output_dir, 'hard_negatives.npy'), self.hard_negatives)
        
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
            'n_ground_truth_per_query': int(self.n_ground_truth_per_query),
            'corpus_size': int(self.corpus_size),
            'random_seed': int(self.random_seed),
            'dataset_type': 'hard_opposing_pairs_with_negatives',
            'strategy': 'opposing_pairs_ground_truth + hard_negatives'
        }
        # Convert config to JSON-serializable format
        config_serializable = convert_to_serializable(config)
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config_serializable, f, indent=4)
        
        print("Data saved successfully!")
        print(f"Files created:")
        print(f"  - corpus.npy: {self.corpus.shape}")
        print(f"  - queries.npy: {self.queries.shape}")
        print(f"  - hard_negatives.npy: {self.hard_negatives.shape}")
        print(f"  - query_ground_truth_pairs.json")
        print(f"  - config.json")
    
    def generate_all(self) -> Dict[str, Any]:
        """
        Generate all components of the challenging synthetic dataset.
        
        Returns:
            Dictionary containing all generated data
        """
        print("=== Starting HARD Synthetic Data Generation ===")
        print(f"Configuration:")
        print(f"  Dimensions: {self.d}")
        print(f"  Training queries: {self.n_train_queries}")
        print(f"  Test queries: {self.n_test_queries}")
        print(f"  Ground truth vectors per query: {self.n_ground_truth_per_query}")
        print(f"  Corpus size: {self.corpus_size}")
        print(f"  Random seed: {self.random_seed}")
        print(f"  Strategy: Opposing pairs + Hard Negatives")
        print()
        
        # Generate all components
        self.generate_queries()
        self.generate_ground_truth_vectors()  # This generates opposing pairs + hard negatives
        self.generate_corpus()
        
        print("\n=== Generation Complete ===")
        print("This dataset is designed to be challenging for single-vector retrieval methods!")
        
        return {
            'queries': self.queries,
            'ground_truth_vectors': self.ground_truth_vectors,
            'hard_negatives': self.hard_negatives,
            'corpus': self.corpus,
            'query2ground_truth_mapping': self.query2ground_truth_mapping
        }


def main():
    parser = argparse.ArgumentParser(description='Generate HARD synthetic data for information retrieval evaluation')
    parser.add_argument('--dimensions', '-d', type=int, default=128, 
                       help='Dimensionality of vectors (default: 128)')
    parser.add_argument('--train-queries', type=int, default=2000,
                       help='Number of training queries (default: 2000)')
    parser.add_argument('--test-queries', type=int, default=200,
                       help='Number of test queries (default: 200)')
    parser.add_argument('--ground-truth-per-query', type=int, default=5,
                       help='Number of ground truth vectors per query (default: 5)')
    parser.add_argument('--corpus-size', type=int, default=100000,
                       help='Total corpus size (default: 100000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', '-o', type=str, default='./hard_synthetic_data',
                       help='Output directory (default: ./hard_synthetic_data)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = HardSyntheticDataGenerator(
        d=args.dimensions,
        n_train_queries=args.train_queries,
        n_test_queries=args.test_queries,
        n_ground_truth_per_query=args.ground_truth_per_query,
        corpus_size=args.corpus_size,
        random_seed=args.seed
    )
    
    # Generate all data
    data = generator.generate_all()
    
    # Save data
    generator.save_data(args.output_dir)
    
    print(f"\n🎯 Dataset generated! Expected to be MUCH HARDER than rotation-based baseline.")
    print(f"   Run baseline evaluation to see the difference:")
    print(f"   python baseline_evaluation.py --data-dir {args.output_dir}")


if __name__ == '__main__':
    main() 