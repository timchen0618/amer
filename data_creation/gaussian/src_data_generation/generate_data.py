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


class SyntheticDataGenerator:
    """
    Generator for synthetic information retrieval dataset with Gaussian queries
    and rotated ground truth vectors.
    """
    
    def __init__(self, 
                 d: int = 128,
                 n_train_queries: int = 2000,
                 n_test_queries: int = 200,
                 n_rotations: int = 5,
                 corpus_size: int = 100000,
                 random_seed: int = 42):
        """
        Initialize the synthetic data generator.
        
        Args:
            d: Dimensionality of vectors
            n_train_queries: Number of training queries
            n_test_queries: Number of test queries  
            n_rotations: Number of rotation matrices (K)
            corpus_size: Total size of the corpus
            random_seed: Random seed for reproducibility
        """
        self.d = d
        self.n_train_queries = n_train_queries
        self.n_test_queries = n_test_queries
        self.n_total_queries = n_train_queries + n_test_queries
        self.n_rotations = n_rotations
        self.corpus_size = corpus_size
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
        
        # Will store generated data
        self.queries = None
        self.rotation_matrices = None
        self.ground_truth_vectors = None
        self.corpus = None
        self.ground_truth_indices = None
        
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
    
    def generate_rotation_matrices(self) -> List[np.ndarray]:
        """
        Generate K orthogonal rotation matrices that are sufficiently different.
        Uses QR decomposition of random matrices to ensure orthogonality.
        
        Returns:
            List of K rotation matrices, each of shape (d, d)
        """
        print(f"Generating {self.n_rotations} rotation matrices...")
        
        rotation_matrices = []
        max_attempts = 1000
        min_angle_threshold = np.pi / 6  # 30 degrees minimum separation
        
        for i in range(self.n_rotations):
            attempts = 0
            while attempts < max_attempts:
                # Generate random matrix and get orthogonal matrix via QR decomposition
                random_matrix = np.random.randn(self.d, self.d)
                q, r = qr(random_matrix)
                
                # Ensure determinant is 1 (proper rotation, not reflection)
                if np.linalg.det(q) < 0:
                    q[:, 0] *= -1
                
                # Check if this matrix is sufficiently different from existing ones
                if self._is_sufficiently_different(q, rotation_matrices, min_angle_threshold):
                    rotation_matrices.append(q)
                    break
                    
                attempts += 1
            
            if attempts == max_attempts:
                print(f"Warning: Could not find sufficiently different matrix {i+1}, using current candidate")
                rotation_matrices.append(q)
        
        self.rotation_matrices = rotation_matrices
        print(f"Generated {len(rotation_matrices)} rotation matrices")
        self._print_matrix_angles()
        return rotation_matrices
    
    def _is_sufficiently_different(self, candidate: np.ndarray, 
                                 existing_matrices: List[np.ndarray], 
                                 min_angle: float) -> bool:
        """
        Check if a candidate rotation matrix is sufficiently different from existing ones.
        """
        if not existing_matrices:
            return True
            
        for existing in existing_matrices:
            # Calculate the angle between rotation matrices using trace formula
            # For rotation matrices A and B: cos(angle) = (trace(A^T @ B) - 1) / 2
            trace_val = np.trace(candidate.T @ existing)
            # Clamp to valid range for arccos due to numerical precision
            cos_angle = np.clip((trace_val - 1) / 2, -1, 1)
            angle = np.arccos(cos_angle)
            
            if angle < min_angle:
                return False
        return True
    
    def _print_matrix_angles(self):
        """Print angles between all pairs of rotation matrices for verification."""
        print("Angles between rotation matrices (in degrees):")
        for i in range(len(self.rotation_matrices)):
            for j in range(i + 1, len(self.rotation_matrices)):
                trace_val = np.trace(self.rotation_matrices[i].T @ self.rotation_matrices[j])
                cos_angle = np.clip((trace_val - 1) / 2, -1, 1)
                angle_deg = np.degrees(np.arccos(cos_angle))
                print(f"  Matrix {i+1} <-> Matrix {j+1}: {angle_deg:.1f}°")
    
    def generate_ground_truth_vectors(self) -> np.ndarray:
        """
        Generate ground truth vectors by applying rotation matrices to queries.
        For each query x, creates y_i = A_i @ x for i=1,...,K
        
        Returns:
            Array of shape (n_total_queries * n_rotations, d)
        """
        if self.queries is None or self.rotation_matrices is None:
            raise ValueError("Must generate queries and rotation matrices first")
            
        print(f"Generating ground truth vectors...")
        
        ground_truth_vectors = []
        for i, rotation_matrix in enumerate(self.rotation_matrices):
            # Apply rotation to all queries
            rotated_queries = self.queries @ rotation_matrix.T
            ground_truth_vectors.append(rotated_queries)
        
        # Stack all ground truth vectors
        self.ground_truth_vectors = np.vstack(ground_truth_vectors)
        print(f"Generated {self.ground_truth_vectors.shape[0]} ground truth vectors")
        return self.ground_truth_vectors
    
    def generate_corpus(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the full corpus by combining ground truth vectors with random vectors.
        
        Returns:
            Tuple of (corpus, ground_truth_indices)
            - corpus: Array of shape (corpus_size, d)
            - ground_truth_indices: Array indicating which corpus vectors are ground truth
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

        # create mapping from query index to ground truth index in the corpus
        query2ground_truth_mapping = {}
        for i in range(self.n_total_queries):
            query2ground_truth_mapping[i] = [i*self.n_rotations + j for j in range(self.n_rotations)]
        
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
            
            # # Find ground truth corpus indices for this query
            # gt_corpus_indices = []
            # for rot_idx in range(self.n_rotations):
            #     # Fix: Ground truth vectors are organized as [rot0_allqueries, rot1_allqueries, ...]
            #     # So for query_idx and rot_idx, the original index is: rot_idx * n_total_queries + query_idx
            #     original_gt_idx = rot_idx * self.n_total_queries + query_idx
            #     if original_gt_idx in self.ground_truth_mapping:
            #         corpus_idx = self.ground_truth_mapping[original_gt_idx]
            #         gt_corpus_indices.append(int(corpus_idx))  # Convert to native int
            
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
                'n_rotations': int(self.n_rotations),
                'dimensions': int(self.d),
                'corpus_size': int(self.corpus_size)
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
        
        # Save rotation matrices
        rotation_matrices_array = np.array(self.rotation_matrices)
        np.save(os.path.join(output_dir, 'rotation_matrices.npy'), rotation_matrices_array)
        
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
            'n_rotations': int(self.n_rotations),
            'corpus_size': int(self.corpus_size),
            'random_seed': int(self.random_seed)
        }
        # Convert config to JSON-serializable format
        config_serializable = convert_to_serializable(config)
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config_serializable, f, indent=4)
        
        print("Data saved successfully!")
        print(f"Files created:")
        print(f"  - corpus.npy: {self.corpus.shape}")
        print(f"  - queries.npy: {self.queries.shape}")
        print(f"  - rotation_matrices.npy: {rotation_matrices_array.shape}")
        print(f"  - query_ground_truth_pairs.json")
        print(f"  - config.json")
    
    def generate_all(self) -> Dict[str, Any]:
        """
        Generate all components of the synthetic dataset.
        
        Returns:
            Dictionary containing all generated data
        """
        print("=== Starting Synthetic Data Generation ===")
        print(f"Configuration:")
        print(f"  Dimensions: {self.d}")
        print(f"  Training queries: {self.n_train_queries}")
        print(f"  Test queries: {self.n_test_queries}")
        print(f"  Rotation matrices: {self.n_rotations}")
        print(f"  Corpus size: {self.corpus_size}")
        print(f"  Random seed: {self.random_seed}")
        print()
        
        # Generate all components
        self.generate_queries()
        self.generate_rotation_matrices()
        self.generate_ground_truth_vectors()
        self.generate_corpus()
        
        print("\n=== Generation Complete ===")
        
        return {
            'queries': self.queries,
            'rotation_matrices': self.rotation_matrices,
            'ground_truth_vectors': self.ground_truth_vectors,
            'corpus': self.corpus,
            'ground_truth_indices': self.ground_truth_indices
        }


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic data for information retrieval evaluation')
    parser.add_argument('--dimensions', '-d', type=int, default=128, 
                       help='Dimensionality of vectors (default: 128)')
    parser.add_argument('--train-queries', type=int, default=2000,
                       help='Number of training queries (default: 2000)')
    parser.add_argument('--test-queries', type=int, default=200,
                       help='Number of test queries (default: 200)')
    parser.add_argument('--rotations', '-k', type=int, default=5,
                       help='Number of rotation matrices (default: 5)')
    parser.add_argument('--corpus-size', type=int, default=100000,
                       help='Total corpus size (default: 100000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', '-o', type=str, default='./synthetic_data',
                       help='Output directory (default: ./synthetic_data)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = SyntheticDataGenerator(
        d=args.dimensions,
        n_train_queries=args.train_queries,
        n_test_queries=args.test_queries,
        n_rotations=args.rotations,
        corpus_size=args.corpus_size,
        random_seed=args.seed
    )
    
    # Generate all data
    data = generator.generate_all()
    
    # Save data
    generator.save_data(args.output_dir)


if __name__ == '__main__':
    main()
