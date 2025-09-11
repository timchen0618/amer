#!/usr/bin/env python3
"""
Highly Diverse MLPs Synthetic Data Generator for Information Retrieval Evaluation

This generator creates a synthetic dataset using maximally different MLP transformations:
- Linear-like MLP (minimal non-linearity, tiny hidden layer)
- High-variance MLP (strong non-linearity, large weights)  
- Bottleneck MLP (severe compression, tiny hidden layer)
- Uniform Distribution MLP (balanced random weights)
- Extreme Sparse MLP (70% zeros, scaled remaining weights)

This creates maximally different transformation behaviors for varied retrieval challenges.
"""
import random
import numpy as np
import json
import os
from typing import Tuple, List, Dict, Any
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
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

def save_corpus(corpus, output_dir: str):
    """
    Save the corpus to a file. Corpus is a numpy array of shape (n_corpus, d)
    """
    allids = np.arange(corpus.shape[0])
    save_file = os.path.join(output_dir, "passages_00")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving {len(corpus)} passage embeddings to {save_file}.")
    with open(save_file, mode="wb") as f:
        pickle.dump((allids, corpus), f)

    print(f"Total passages processed {len(allids)}. Written to {save_file}.")


class TransformationMLP(nn.Module):
    """
    Multi-Layer Perceptron for vector transformation.
    """
    def __init__(self, input_dim: int, hidden_dim: int = None, num_layers: int = 2):
        super(TransformationMLP, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim
            
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)


class OpposingPairsMlpSyntheticDataGenerator:
    """
    Generator for synthetic information retrieval dataset with Gaussian queries
    and highly diverse ground truth vectors using MLPs.
    
    Uses 5 maximally different MLP transformations:
    - Linear-like MLP (minimal non-linearity)
    - High-variance MLP (strong non-linearity)
    - Bottleneck MLP (severe compression)
    - Uniform Distribution MLP (balanced weights)
    - Extreme Sparse MLP (70% sparsity)
    
    This creates maximally different transformation behaviors for varied retrieval challenges.
    """
    
    def __init__(self, 
                 d: int = 128,
                 n_train_queries: int = 2000,
                 n_test_queries: int = 200,
                 n_transformations: int = 5,  # Fixed to 5 for diverse MLPs
                 corpus_size: int = 100000,
                 hidden_dim: int = None,
                 num_layers: int = 2,
                 random_seed: int = 42,
                 multiple_query_distributions: bool = False,
                 ood_distribution: bool = False,
                 hard_type: str = 'opposite'):
        """
        Initialize the diverse MLPs synthetic data generator.
        
        Args:
            d: Dimensionality of vectors
            n_train_queries: Number of training queries
            n_test_queries: Number of test queries  
            n_transformations: Number of MLPs (fixed to 5)
            corpus_size: Total size of the corpus
            hidden_dim: Hidden dimension for MLPs (default: same as d)
            num_layers: Number of layers in each MLP
            random_seed: Random seed for reproducibility
            multiple_query_distributions: Whether to use multiple query distributions
            ood_distribution: Whether to use OOD distribution (training: first 4 dists, testing: last dist)
            hard_type: Type of hard transformation (default: opposite), choices=['opposite', 'rotation', 'normal']
        """
        self.d = d
        self.n_train_queries = n_train_queries
        self.n_test_queries = n_test_queries
        self.n_total_queries = n_train_queries + n_test_queries
        self.n_transformations = n_transformations  # Always 5 for diverse MLPs approach
        self.corpus_size = corpus_size
        self.hidden_dim = hidden_dim if hidden_dim is not None else d
        self.num_layers = num_layers
        self.random_seed = random_seed
        self.multiple_query_distributions = multiple_query_distributions
        self.ood_distribution = ood_distribution
        self.hard_type = hard_type
        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Validation
        if self.ood_distribution and not self.multiple_query_distributions:
            raise ValueError("ood_distribution requires multiple_query_distributions to be enabled")
        
        # Will store generated data
        self.queries = None
        self.transformation_mlps = None
        self.ground_truth_vectors = None
        self.corpus = None
        self.query2ground_truth_mapping = None
        
    def _generate_standard_gaussian(self, n_queries: int) -> np.ndarray:
        """Generate queries from standard Gaussian N(0, I)"""
        return np.random.multivariate_normal(
            mean=np.zeros(self.d),
            cov=np.eye(self.d),
            size=n_queries
        )
    
    def _generate_highvar_gaussian(self, n_queries: int) -> np.ndarray:
        """Generate queries from high-variance Gaussian N(0, 4I)"""
        return np.random.multivariate_normal(
            mean=np.zeros(self.d),
            cov=4.0 * np.eye(self.d),
            size=n_queries
        )
    
    def _generate_correlated_gaussian(self, n_queries: int) -> np.ndarray:
        """Generate queries from correlated Gaussian with random covariance"""
        A = np.random.randn(self.d, self.d)
        cov_matrix = 0.5 * (A @ A.T) + 0.1 * np.eye(self.d)
        return np.random.multivariate_normal(
            mean=np.zeros(self.d),
            cov=cov_matrix,
            size=n_queries
        )
    
    def _generate_uniform(self, n_queries: int) -> np.ndarray:
        """Generate queries from uniform distribution [-2, 2]^d"""
        return np.random.uniform(
            low=-2.0, high=2.0, size=(n_queries, self.d)
        )
    
    def _generate_laplace_gaussian(self, n_queries: int) -> np.ndarray:
        """Generate queries from Laplace + Gaussian noise"""
        laplace_queries = np.random.laplace(
            loc=0.0, scale=1.0, size=(n_queries, self.d)
        )
        gaussian_noise = np.random.normal(0, 0.1, size=(n_queries, self.d))
        return laplace_queries + gaussian_noise
    
    def _generate_shifted_uniform(self, n_queries: int) -> np.ndarray:
        """Generate queries from uniform distribution [0, 4]^d"""
        return np.random.uniform(
            low=0.0, high=4.0, size=(n_queries, self.d)
        )

    def generate_queries(self) -> np.ndarray:
        """
        Generate query vectors from multivariate Gaussian N(0, I) or multiple distributions.
        
        Returns:
            Array of shape (n_total_queries, d)
        """
        if self.multiple_query_distributions:
            print(f"Generating {self.n_total_queries} query vectors from multiple distributions...")
            
            # Define distribution generators and names
            generators = [
                (self._generate_standard_gaussian, "Standard Gaussian N(0, I)"),
                (self._generate_highvar_gaussian, "High-variance Gaussian N(0, 4I)"),
                (self._generate_correlated_gaussian, "Correlated Gaussian with random covariance"),
                (self._generate_uniform, "Uniform distribution [-2, 2]^d"),
                (self._generate_laplace_gaussian, "Laplace + Gaussian noise")
            ]
            
            if self.ood_distribution:
                # OOD mode: first 4 distributions for training, last 1 for testing
                print(f"🔄 Using OOD distribution split...")
                print(f"Target: {self.n_train_queries} training + {self.n_test_queries} testing")
                
                # Training distributions (first 4)
                train_queries_per_dist = self.n_train_queries // 4
                train_remainder = self.n_train_queries % 4
                
                train_queries = []
                for i, (generator, name) in enumerate(generators[:4]):
                    n_queries = train_queries_per_dist + (1 if i < train_remainder else 0)
                    queries = generator(n_queries)
                    train_queries.append(queries)
                    print(f"  ✓ Generated {n_queries} TRAINING queries from {name}")
                
                # Test distribution (last 1)
                test_queries = []
                generator, name = generators[4]
                queries = generator(self.n_test_queries)
                test_queries.append(queries)
                print(f"  ✓ Generated {self.n_test_queries} TESTING queries from {name} (OOD)")
                
                # Combine and shuffle within each split
                train_combined = np.vstack(train_queries)
                test_combined = np.vstack(test_queries)
                
                train_indices = np.random.permutation(len(train_combined))
                test_indices = np.random.permutation(len(test_combined))
                train_combined = train_combined[train_indices]
                test_combined = test_combined[test_indices]
                
                # Final combination (training first, then testing)
                queries = np.vstack([train_combined, test_combined])
                self.n_total_queries = len(queries)
                
                print(f"  ✅ Final split: {len(train_combined)} training + {len(test_combined)} testing = {self.n_total_queries} total")
                
            else:
                # Regular mode: all distributions mixed equally
                queries_per_dist = self.n_total_queries // 5
                remaining_queries = self.n_total_queries % 5
                
                all_queries = []
                for i, (generator, name) in enumerate(generators):
                    n_queries = queries_per_dist + (1 if i < remaining_queries else 0)
                    queries = generator(n_queries)
                    all_queries.append(queries)
                    print(f"  ✓ Generated {n_queries} queries from {name}")
                
                # Combine all queries and shuffle
                queries = np.vstack(all_queries)
                shuffle_indices = np.random.permutation(len(queries))
                queries = queries[shuffle_indices]
                
                print(f"  → Total: {queries.shape[0]} queries from 5 different distributions (shuffled)")
            
        else:
            if self.n_transformations == 5:
                print(f"Generating {self.n_total_queries} query vectors of dimension {self.d}...standard gaussian")
                queries = self._generate_standard_gaussian(self.n_total_queries)
            elif self.n_transformations == 2:
                print(f"Generating {self.n_total_queries} query vectors of dimension {self.d}...shifted uniform")
                queries = self._generate_shifted_uniform(self.n_total_queries)
                # queries = self._generate_uniform(self.n_total_queries)
            
        self.queries = queries
        return queries
    
    def generate_rotation(self):
        import torch

        def proj_to_SO(X):
            # Polar decomposition: nearest orthogonal matrix in Frobenius norm
            U, S, Vt = torch.linalg.svd(X, full_matrices=False)
            Q = U @ Vt
            # Fix det = +1 (proper rotation)
            if torch.det(Q) < 0:
                U[:, -1] *= -1
                Q = U @ Vt
            return Q

        def frob_gram_schmidt(X, basis):
            # Make X Frobenius-orthogonal to every matrix in 'basis'
            for B in basis:
                num = torch.trace(B.T @ X)
                den = torch.trace(B.T @ B)
                X = X - (num / den) * B
            return X

        def rotation_orthogonal_to(basis, n, iters=20):
            # basis: list of rotation matrices you want orthogonality to (Frobenius)
            # returns a rotation C with <C, Bi>_F ≈ 0 for all Bi in basis
            X = torch.randn(n, n)
            for _ in range(iters):
                X = frob_gram_schmidt(X, basis)  # enforce ⟨X, Bi⟩_F = 0 in matrix space
                X = proj_to_SO(X)                # retract to SO(n)
            # one last clean-up
            X = frob_gram_schmidt(X, basis)
            X = proj_to_SO(X)
            return X

        def random_rotation(n):
            A = torch.randn(n, n)
            return proj_to_SO(A)

        # Example: build A, then B ⟂ A, then C ⟂ {A,B}
        n = 1024
        A = random_rotation(n)
        B = rotation_orthogonal_to([A], n, iters=20)
        C = rotation_orthogonal_to([A, B], n, iters=20)
        # D = rotation_orthogonal_to([A, B, C], n, iters=20)
        # E = rotation_orthogonal_to([A, B, C, D], n, iters=20)
        return A, B, C
    
    def generate_normal(self):
        # get a random rotation matrix of size d x d; make it norm 1
        rotation_matrix = np.random.randn(self.d, self.d)
        rotation_matrix = rotation_matrix / np.linalg.norm(rotation_matrix)
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32)
        
        # get another rotation matrix of size d x d that is orthogonal to the first one
        rotation_matrix2 = np.random.randn(self.d, self.d)
        rotation_matrix2 = rotation_matrix2 / np.linalg.norm(rotation_matrix2)
        rotation_matrix2 = torch.tensor(rotation_matrix2, dtype=torch.float32)
        rotation_matrix2 = rotation_matrix2.T @ rotation_matrix
        rotation_matrix2 = rotation_matrix2 / np.linalg.norm(rotation_matrix2)
        rotation_matrix2 = torch.tensor(rotation_matrix2, dtype=torch.float32)
        
        # get a random matrix of size d x d that is orthogonal to the first two
        rotation_matrix3 = np.random.randn(self.d, self.d)
        rotation_matrix3 = rotation_matrix3 / np.linalg.norm(rotation_matrix3)
        rotation_matrix3 = torch.tensor(rotation_matrix3, dtype=torch.float32)
        rotation_matrix3 = rotation_matrix3.T @ rotation_matrix
        rotation_matrix3 = rotation_matrix3 / np.linalg.norm(rotation_matrix3)
        rotation_matrix3 = torch.tensor(rotation_matrix3, dtype=torch.float32)
        rotation_matrix3 = rotation_matrix3.T @ rotation_matrix2
        rotation_matrix3 = rotation_matrix3 / np.linalg.norm(rotation_matrix3)
        rotation_matrix3 = torch.tensor(rotation_matrix3, dtype=torch.float32)
        
        return rotation_matrix, rotation_matrix2, rotation_matrix3
    
    
    def generate_transformation_mlps(self) -> List[TransformationMLP]:
        """
        Generate 5 distinctly different MLP transformations with very diverse behaviors.
        This creates varied transformations: [A, B, C, D, E] with maximally different characteristics.
        
        Returns:
            List of 5 MLPs
        """
        print(f"Generating highly diverse MLP transformations...")
        
        transformation_mlps = []
        
        if self.n_transformations == 5:
            
            if self.hard_type in ['opposite', 'rotation']:
                rotation_matrix, rotation_matrix2, rotation_matrix3 = self.generate_rotation()
                if self.hard_type == 'rotation':
                    rotation_matrix4 = -rotation_matrix2
                    rotation_matrix5 = -rotation_matrix3
            else:
                rotation_matrix, rotation_matrix2, rotation_matrix3 = self.generate_normal()
                rotation_matrix4 = -rotation_matrix2
                rotation_matrix5 = -rotation_matrix3
        elif self.n_transformations == 2:
            rotation_matrix, _, _ = self.generate_normal()
            rotation_matrix2 = -rotation_matrix

        # 1. Linear-like MLP (small hidden layer, minimal non-linearity)
        mlp1 = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.GELU(),
            nn.Linear(self.d, self.d),
            nn.GELU(),
            nn.Linear(self.d, self.d)
        )
        with torch.no_grad():
            for param in mlp1.parameters():
                if len(param.shape) > 1:
                    # Very small weights to minimize non-linear effects
                    # nn.init.uniform_(param, -4, 0)
                    param.data = rotation_matrix
                else:
                    nn.init.zeros_(param)
        transformation_mlps.append(mlp1)
        print(f"  ✓ Added Linear-like MLP (A) - minimal non-linearity")
        print(f"    Architecture: {self.d} → {self.hidden_dim} → {self.d}")
        print(f"    Hidden dim ratio: {self.hidden_dim/self.d:.3f}")
        print(f"    Initialization: Normal(0, 0.01) for weights, Normal(0, 0.05) for biases")
        
        # 2. High-variance MLP (large weights, strong non-linearity)
        mlp2 = nn.Sequential(
            nn.Linear(self.d, self.d),
            nn.GELU(),
            nn.Linear(self.d, self.d),
            nn.GELU(),
            nn.Linear(self.d, self.d)
        )
        with torch.no_grad():
            for param in mlp2.parameters():
                if len(param.shape) > 1:
                    # Large variance to create strong non-linear effects
                    # nn.init.uniform_(param, -2, 2)
                    param.data = rotation_matrix2
                else:
                    nn.init.zeros_(param)
        transformation_mlps.append(mlp2)
        print(f"  ✓ Added High-variance MLP (B) - strong non-linearity")
        print(f"    Architecture: {self.d} → {self.hidden_dim} → {self.d}")
        print(f"    Hidden dim ratio: {self.hidden_dim/self.d:.3f}")
        print(f"    Initialization: Normal(0, 1.0) for weights, Normal(0, 0.5) for biases")
        
        if self.n_transformations == 5:
            # 3. Bottleneck MLP (very small hidden layer for compression)
            mlp3 = nn.Sequential(
                nn.Linear(self.d, self.d),
                nn.GELU(),
                nn.Linear(self.d, self.d),
                nn.GELU(),
                nn.Linear(self.d, self.d)
            )
            with torch.no_grad():
                for param in mlp3.parameters():
                    if len(param.shape) > 1:
                        # nn.init.uniform_(param, -1, 1)
                        param.data = rotation_matrix3
                    else:
                        nn.init.zeros_(param)
            transformation_mlps.append(mlp3)
            print(f"  ✓ Added Bottleneck MLP (C) - severe compression")
            print(f"    Architecture: {self.d} → {self.hidden_dim} → {self.d}")
            print(f"    Hidden dim ratio: {self.hidden_dim/self.d:.3f}")
            print(f"    Initialization: Normal(1, 2) for weights, Normal(1, 2) for biases")
            
            # 4. Uniform Distribution MLP (as before)
            if self.hard_type == 'opposite':
                mlp4 = mlp2
            else:
                mlp4 = nn.Sequential(
                    nn.Linear(self.d, self.d),
                    nn.GELU(),
                    nn.Linear(self.d, self.d),
                    nn.GELU(),
                    nn.Linear(self.d, self.d)
                )
                
            with torch.no_grad():
                for param in mlp4.parameters():
                    if len(param.shape) > 1:
                        # nn.init.uniform_(param, 0, 2)
                        param.data = rotation_matrix4
                    else:
                        nn.init.zeros_(param)
            transformation_mlps.append(mlp4)
            print(f"  ✓ Added Uniform Distribution MLP (D)")
            print(f"    Architecture: {self.d} → {self.d} → {self.d}")
            print(f"    Hidden dim ratio: {self.d/self.d:.3f}")
            print(f"    Initialization: Uniform(-2, 2) for weights, Uniform(-1, 1) for biases")
            
            # 5. Extreme Sparse MLP (70% weights are zero)
            if self.hard_type == 'opposite':
                mlp5 = mlp3
            else:
                mlp5= nn.Sequential(
                    nn.Linear(self.d, self.d),
                    nn.GELU(),
                    nn.Linear(self.d, self.d),
                    nn.GELU(),
                    nn.Linear(self.d, self.d)
                )
            
            with torch.no_grad():
                for param in mlp5.parameters():
                    if len(param.shape) > 1:
                        # nn.init.uniform_(param, 0, 2)
                        param.data = rotation_matrix5
                    else:
                        nn.init.zeros_(param)
            transformation_mlps.append(mlp5)
            print(f"  ✓ Added Sigmoid MLP (E) - strong non-linearity")
            print(f"    Architecture: {self.d} → {self.d} → {self.d} → {self.d}")
            print(f"    Hidden dim ratio: {self.d/self.d:.3f}")
            print(f"    Initialization: Normal(0, 10.0) for weights, Normal(0, 10.0) for biases")
        
        # Print parameter counts
        print(f"\nParameter counts:")
        for i, mlp in enumerate(transformation_mlps):
            total_params = sum(p.numel() for p in mlp.parameters())
            trainable_params = sum(p.numel() for p in mlp.parameters() if p.requires_grad)
            print(f"  MLP {i+1}: {total_params:,} total parameters ({trainable_params:,} trainable)")
        
        # Set to evaluation mode
        for mlp in transformation_mlps:
            mlp.eval()
        
        self.transformation_mlps = transformation_mlps
        print(f"\nGenerated {len(transformation_mlps)} highly diverse MLP transformations")
        self._print_mlp_relationships()
        self._test_diversity_effect()
        return transformation_mlps
    
    def _print_mlp_relationships(self):
        """Print relationships between MLPs."""
        print("\nMLP transformation structure:")
        print("  MLP 1: Linear-like (tiny hidden layer, minimal non-linearity)")
        print("  MLP 2: High-variance (large weights, strong non-linearity)")
        print("  MLP 3: Bottleneck (severe compression through tiny hidden layer)") 
        print("  MLP 4: Uniform distribution (balanced random weights)")
        print("  MLP 5: Extreme sparse (70% zeros, scaled up remaining weights)")
        print("  → Maximally different architectures and initialization strategies")
    
    def _test_diversity_effect(self):
        """Test the diversity effect of different MLPs."""
        print("\nTesting MLP diversity effect:")
        
        # Generate a sample query
        sample_query = np.random.randn(self.d)
        sample_query = sample_query / np.linalg.norm(sample_query)
        sample_query_tensor = torch.tensor(sample_query, dtype=torch.float32).unsqueeze(0)
        
        # Apply all MLPs
        transformed_vectors = []
        with torch.no_grad():
            for i, mlp in enumerate(self.transformation_mlps):
                transformed = mlp(sample_query_tensor).squeeze(0).numpy()
                transformed_vectors.append(transformed)
        
        # Compute pairwise similarities between transformed vectors
        print("  Pairwise similarities between MLP outputs:")
        for i in range(len(transformed_vectors)):
            for j in range(i+1, len(transformed_vectors)):
                vec_i = transformed_vectors[i]
                vec_j = transformed_vectors[j]
                sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))
                print(f"    MLP_{i+1} vs MLP_{j+1}: {sim:.3f}")
        
        # Compute average
        average_vector = np.mean(transformed_vectors, axis=0)
        
        # Compute similarities with average
        similarities = []
        for i, vec in enumerate(transformed_vectors):
            sim = np.dot(average_vector, vec) / (np.linalg.norm(average_vector) * np.linalg.norm(vec))
            similarities.append(sim)
            print(f"  Similarity(avg, MLP_{i+1}): {sim:.3f}")
        
        avg_similarity = np.mean(similarities)
        print(f"  Average similarity with mean: {avg_similarity:.3f}")
        print(f"  Diversity level: {'HIGH' if avg_similarity < 0.7 else 'MEDIUM' if avg_similarity < 0.85 else 'LOW'}")
    
    def generate_ground_truth_vectors(self) -> np.ndarray:
        """
        Generate ground truth vectors by applying diverse MLPs to queries.
        For each query x, creates y_i = MLP_i(x) for i=1,...,5 where MLPs have diverse initializations
        
        Returns:
            Array of shape (n_total_queries * n_transformations, d)
        """
        if self.queries is None or self.transformation_mlps is None:
            raise ValueError("Must generate queries and MLPs first")
            
        print(f"Generating ground truth vectors using diverse MLPs...")
        
        ground_truth_vectors = []
        
        # Convert queries to tensor
        queries_tensor = torch.tensor(self.queries, dtype=torch.float32)
        
        # Generate ground truth vectors organized by query:
        # [Q0MLP0, Q0MLP1, Q0MLP2, Q0MLP3, Q0MLP4, Q1MLP0, Q1MLP1, Q1MLP2, Q1MLP3, Q1MLP4, ...]
        with torch.no_grad():
            for query_idx, query in enumerate(tqdm(queries_tensor)):
                query_batch = query.unsqueeze(0)  # Add batch dimension
                for mlp_idx, mlp in enumerate(self.transformation_mlps[:self.n_transformations]):
                    # Apply MLP to this specific query
                    if self.hard_type == 'opposite' and mlp_idx >= 3:
                        transformed_query = -mlp(query_batch).squeeze(0).numpy()
                    else:
                        transformed_query = mlp(query_batch).squeeze(0).numpy()
                    ground_truth_vectors.append(transformed_query)
        
        # Convert to numpy array
        self.ground_truth_vectors = np.array(ground_truth_vectors)
        print(f"Generated {self.ground_truth_vectors.shape[0]} ground truth vectors")
        print(f"Organization: [Q0MLP0, Q0MLP1, ..., Q0MLP4, Q1MLP0, Q1MLP1, ..., Q1MLP4, ...]")
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
                'hidden_dim': int(self.hidden_dim),
                'num_layers': int(self.num_layers),
                'transformation_type': 'highly_diverse_mlps'
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
        
        # Save MLPs
        mlps_dir = os.path.join(output_dir, 'mlps')
        os.makedirs(mlps_dir, exist_ok=True)
        for i, mlp in enumerate(self.transformation_mlps):
            torch.save(mlp.state_dict(), os.path.join(mlps_dir, f'mlp_{i}.pth'))
        
        # Save MLP architecture info
        mlp_config = {
            'input_dim': self.d,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'n_mlps': len(self.transformation_mlps)
        }
        with open(os.path.join(mlps_dir, 'mlp_config.json'), 'w') as f:
            json.dump(mlp_config, f, indent=4)
        
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
            'hidden_dim': int(self.hidden_dim),
            'num_layers': int(self.num_layers),
            'random_seed': int(self.random_seed),
            'multiple_query_distributions': bool(self.multiple_query_distributions),
            'ood_distribution': bool(self.ood_distribution),
            'transformation_type': 'highly_diverse_mlps',
            'transformations': 'linear_like + high_variance + bottleneck + uniform + extreme_sparse',
            'approach': 'Linear_MLP + HighVar_MLP + Bottleneck_MLP + Uniform_MLP + ExtremeSparse_MLP'
        }
        # Convert config to JSON-serializable format
        config_serializable = convert_to_serializable(config)
        with open(os.path.join(output_dir, 'config.json'), 'w') as f:
            json.dump(config_serializable, f, indent=4)
        
        print("Data saved successfully!")
        print(f"Files created:")
        print(f"  - corpus.npy: {self.corpus.shape}")
        print(f"  - queries.npy: {self.queries.shape}")
        print(f"  - mlps/: {len(self.transformation_mlps)} MLP state dicts")
        print(f"  - query_ground_truth_pairs.json")
        print(f"  - config.json")
    
    def generate_all(self) -> Dict[str, Any]:
        """
        Generate all components of the diverse MLPs synthetic dataset.
        
        Returns:
            Dictionary containing all generated data
        """
        print("=== Starting Highly Diverse MLPs Synthetic Data Generation ===")
        print(f"Configuration:")
        print(f"  Dimensions: {self.d}")
        print(f"  Training queries: {self.n_train_queries}")
        print(f"  Test queries: {self.n_test_queries}")
        print(f"  Transformations: {self.n_transformations} MLPs (maximally different)")
        print(f"  Hidden dimension: {self.hidden_dim}")
        print(f"  MLP layers: {self.num_layers}")
        print(f"  Corpus size: {self.corpus_size}")
        print(f"  Random seed: {self.random_seed}")
        print(f"  Approach: Linear-like + High-variance + Bottleneck + Uniform + Extreme-sparse MLPs")
        print()
        
        # Generate all components
        self.generate_queries()
        self.generate_transformation_mlps()
        self.generate_ground_truth_vectors()
        self.generate_corpus()
        
        print("\n=== Generation Complete ===")
        print("This dataset uses maximally different MLPs to create highly varied transformation behaviors!")
        
        return {
            'queries': self.queries,
            'transformation_mlps': self.transformation_mlps,
            'ground_truth_vectors': self.ground_truth_vectors,
            'corpus': self.corpus,
            'query2ground_truth_mapping': self.query2ground_truth_mapping
        }


def main():
    parser = argparse.ArgumentParser(description='Generate highly diverse MLPs synthetic data for information retrieval evaluation')
    parser.add_argument('--dimensions', '-d', type=int, default=1024, 
                       help='Dimensionality of vectors (default: 1024)')
    parser.add_argument('--train-queries', type=int, default=2000,
                       help='Number of training queries (default: 2000)')
    parser.add_argument('--test-queries', type=int, default=200,
                       help='Number of test queries (default: 200)')
    parser.add_argument('--corpus-size', type=int, default=50000,
                       help='Total corpus size (default: 50000)')
    parser.add_argument('--hidden-dim', type=int, default=1024,
                       help='Hidden dimension for MLPs (default: same as dimensions)')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of layers in MLPs (default: 2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--output-dir', '-o', type=str, default='./diverse_mlps_data',
                       help='Output directory (default: ./diverse_mlps_data)')
    parser.add_argument('--multiple-query-distributions', action='store_true',
                       help='Use multiple query distributions instead of just Gaussian')
    parser.add_argument('--ood-distribution', action='store_true',
                       help='Use OOD distribution (training: first 4 dists, testing: last dist)')
    parser.add_argument('--sample-transformations', action='store_true',
                       help='Sample a random subset of transformations for each query')
    parser.add_argument('--n-transformations', type=int, default=5,
                       help='Number of transformations to sample for each query (default: 5)')
    parser.add_argument('--hard-type', type=str, default='opposite',
                       help='Type of hard transformation (default: opposite)', choices=['opposite', 'rotation', 'normal'])
    
    args = parser.parse_args()
    
    # Create generator
    generator = OpposingPairsMlpSyntheticDataGenerator(
        d=args.dimensions,
        n_train_queries=args.train_queries,
        n_test_queries=args.test_queries,
        corpus_size=args.corpus_size,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        random_seed=args.seed,
        multiple_query_distributions=args.multiple_query_distributions,
        ood_distribution=args.ood_distribution,
        n_transformations=args.n_transformations,
        hard_type=args.hard_type
    )
    
    # Generate all data
    data = generator.generate_all()
    
    # Save data
    generator.save_data(args.output_dir)
    
    print(f"\n🎯 Highly diverse MLPs dataset generated!")
    print(f"   MLPs provide maximally different non-linear transformations.")
    print(f"   Run baseline evaluation to compare:")
    print(f"   python baseline_evaluation.py --data-dir {args.output_dir}")

    
    # Save corpus
    corpus = np.load(os.path.join(args.output_dir, 'corpus.npy')) 
    save_corpus(corpus, args.output_dir)


if __name__ == '__main__':
    main() 
    # python generate_data_opposing_mlps.py --dimensions 1024 --train-queries 2000 --test-queries 200 --corpus-size 100000 --seed 42 --output-dir ./highly_diverse_mlps_data 