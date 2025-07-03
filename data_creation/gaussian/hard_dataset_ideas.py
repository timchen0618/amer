#!/usr/bin/env python3
"""
Ideas for creating a harder synthetic dataset that challenges single-vector retrieval.
"""

import numpy as np
from typing import List, Tuple
from scipy.linalg import qr


class HardSyntheticDataGenerator:
    """
    Generator for challenging synthetic IR dataset that's hard for single-vector retrieval.
    """
    
    def __init__(self, d: int = 128, n_queries: int = 1000):
        self.d = d
        self.n_queries = n_queries
        
    def strategy_1_multi_aspect_ground_truth(self, query: np.ndarray) -> List[np.ndarray]:
        """
        Strategy 1: Create ground truth vectors representing different aspects.
        Single vector averaging will poorly represent all aspects.
        """
        gt_vectors = []
        
        # Aspect 1: Slight semantic shift (small rotation)
        theta1 = np.pi / 12  # 15 degrees
        R1 = self._random_rotation_matrix(theta1)
        gt_vectors.append(R1 @ query)
        
        # Aspect 2: Different semantic direction (larger rotation)
        theta2 = np.pi / 4   # 45 degrees  
        R2 = self._random_rotation_matrix(theta2)
        gt_vectors.append(R2 @ query)
        
        # Aspect 3: Orthogonal aspect (90 degree rotation in random plane)
        R3 = self._random_rotation_matrix(np.pi / 2)
        gt_vectors.append(R3 @ query)
        
        # Aspect 4: Scaled version with noise (different magnitude)
        scale = 1.5
        noise = np.random.normal(0, 0.1, self.d)
        gt_vectors.append(scale * query + noise)
        
        # Aspect 5: Mixed with random orthogonal vector
        random_vec = np.random.randn(self.d)
        random_vec = random_vec - np.dot(random_vec, query) * query  # Make orthogonal
        random_vec = random_vec / np.linalg.norm(random_vec)
        mix_weight = 0.7
        gt_vectors.append(mix_weight * query + (1 - mix_weight) * random_vec)
        
        return gt_vectors
    
    def strategy_2_clustered_with_hard_negatives(self, query: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Strategy 2: Create clusters of ground truth + add hard negative vectors.
        """
        # Generate ground truth cluster center (perturbed query)
        cluster_center = query + np.random.normal(0, 0.2, self.d)
        cluster_center = cluster_center / np.linalg.norm(cluster_center)
        
        # Generate 5 ground truth vectors around cluster center
        gt_vectors = []
        for i in range(5):
            noise = np.random.normal(0, 0.1, self.d)
            gt_vec = cluster_center + noise
            gt_vec = gt_vec / np.linalg.norm(gt_vec)
            gt_vectors.append(gt_vec)
        
        # Generate hard negatives (similar but not ground truth)
        hard_negatives = []
        for i in range(10):  # 10 hard negatives per query
            # Create vector that's 70-85% similar to ground truth
            similarity_target = 0.7 + 0.15 * np.random.random()
            
            # Generate random direction
            random_dir = np.random.randn(self.d)
            random_dir = random_dir / np.linalg.norm(random_dir)
            
            # Mix with cluster center to achieve target similarity
            hard_neg = similarity_target * cluster_center + np.sqrt(1 - similarity_target**2) * random_dir
            hard_negatives.append(hard_neg)
        
        return gt_vectors, hard_negatives
    
    def strategy_3_hierarchical_structure(self, query: np.ndarray) -> List[np.ndarray]:
        """
        Strategy 3: Create hierarchical ground truth that can't be captured by single vector.
        """
        gt_vectors = []
        
        # Level 1: Main direction (close to query)
        main_dir = query + np.random.normal(0, 0.05, self.d)
        main_dir = main_dir / np.linalg.norm(main_dir)
        gt_vectors.append(main_dir)
        
        # Level 2: Two orthogonal subdirections
        # Create two vectors orthogonal to main_dir and each other
        random1 = np.random.randn(self.d)
        orthogonal1 = random1 - np.dot(random1, main_dir) * main_dir
        orthogonal1 = orthogonal1 / np.linalg.norm(orthogonal1)
        
        random2 = np.random.randn(self.d)
        orthogonal2 = random2 - np.dot(random2, main_dir) * main_dir
        orthogonal2 = orthogonal2 - np.dot(orthogonal2, orthogonal1) * orthogonal1
        orthogonal2 = orthogonal2 / np.linalg.norm(orthogonal2)
        
        # Mix main direction with orthogonal directions
        weight_main = 0.6
        weight_ortho = 0.4
        
        gt_vectors.append(weight_main * main_dir + weight_ortho * orthogonal1)
        gt_vectors.append(weight_main * main_dir + weight_ortho * orthogonal2)
        
        # Level 3: Specific aspects (more orthogonal components)
        orthogonal3 = np.random.randn(self.d)
        for vec in [main_dir, orthogonal1, orthogonal2]:
            orthogonal3 = orthogonal3 - np.dot(orthogonal3, vec) * vec
        orthogonal3 = orthogonal3 / np.linalg.norm(orthogonal3)
        
        orthogonal4 = np.random.randn(self.d)
        for vec in [main_dir, orthogonal1, orthogonal2, orthogonal3]:
            orthogonal4 = orthogonal4 - np.dot(orthogonal4, vec) * vec
        orthogonal4 = orthogonal4 / np.linalg.norm(orthogonal4)
        
        gt_vectors.append(0.4 * main_dir + 0.3 * orthogonal1 + 0.3 * orthogonal3)
        gt_vectors.append(0.4 * main_dir + 0.3 * orthogonal2 + 0.3 * orthogonal4)
        
        return gt_vectors
    
    def strategy_4_anti_averaging_structure(self, query: np.ndarray) -> List[np.ndarray]:
        """
        Strategy 4: Create ground truth vectors that when averaged give a poor retrieval vector.
        """
        gt_vectors = []
        
        # Create vectors in a "star" pattern - they point in different directions
        # so averaging them gives something in the center that's not close to any of them
        
        for i in range(5):
            # Create rotation matrix for 72 degrees apart (360/5)
            angle = (2 * np.pi * i) / 5
            
            # Create rotation in a random 2D plane
            plane_vec1 = np.random.randn(self.d)
            plane_vec1 = plane_vec1 / np.linalg.norm(plane_vec1)
            
            plane_vec2 = np.random.randn(self.d)
            plane_vec2 = plane_vec2 - np.dot(plane_vec2, plane_vec1) * plane_vec1
            plane_vec2 = plane_vec2 / np.linalg.norm(plane_vec2)
            
            # Rotate query in this plane
            rotated = np.cos(angle) * (query @ plane_vec1) * plane_vec1 + \
                     np.sin(angle) * (query @ plane_vec1) * plane_vec2 + \
                     (query - (query @ plane_vec1) * plane_vec1)
            
            # Add some distance from query
            direction = rotated - query
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
                gt_vec = query + 0.8 * direction  # Move away from query
                gt_vectors.append(gt_vec / np.linalg.norm(gt_vec))
            else:
                # Fallback: random direction
                random_dir = np.random.randn(self.d)
                gt_vectors.append(random_dir / np.linalg.norm(random_dir))
        
        return gt_vectors
    
    def _random_rotation_matrix(self, angle: float) -> np.ndarray:
        """Generate random rotation matrix with given angle."""
        # Create random rotation in a random 2D plane
        plane_vec1 = np.random.randn(self.d)
        plane_vec1 = plane_vec1 / np.linalg.norm(plane_vec1)
        
        plane_vec2 = np.random.randn(self.d)
        plane_vec2 = plane_vec2 - np.dot(plane_vec2, plane_vec1) * plane_vec1
        plane_vec2 = plane_vec2 / np.linalg.norm(plane_vec2)
        
        # Create rotation matrix
        R = np.eye(self.d)
        R = R + (np.cos(angle) - 1) * (np.outer(plane_vec1, plane_vec1) + np.outer(plane_vec2, plane_vec2))
        R = R + np.sin(angle) * (np.outer(plane_vec1, plane_vec2) - np.outer(plane_vec2, plane_vec1))
        
        return R


def test_strategies():
    """Test and compare different strategies."""
    generator = HardSyntheticDataGenerator(d=128)
    
    # Test query
    query = np.random.randn(128)
    query = query / np.linalg.norm(query)
    
    print("=== Testing Different Hardness Strategies ===\n")
    
    # Strategy 1: Multi-aspect
    gt_vecs_1 = generator.strategy_1_multi_aspect_ground_truth(query)
    avg_1 = np.mean(gt_vecs_1, axis=0)
    sims_1 = [np.dot(avg_1, gt) for gt in gt_vecs_1]
    print(f"Strategy 1 - Multi-aspect:")
    print(f"  Average similarities to GT: {np.mean(sims_1):.3f} ± {np.std(sims_1):.3f}")
    print(f"  Min similarity: {np.min(sims_1):.3f}")
    
    # Strategy 2: Clustered
    gt_vecs_2, hard_negs_2 = generator.strategy_2_clustered_with_hard_negatives(query)
    avg_2 = np.mean(gt_vecs_2, axis=0)
    sims_2 = [np.dot(avg_2, gt) for gt in gt_vecs_2]
    hard_neg_sims_2 = [np.dot(avg_2, hn) for hn in hard_negs_2]
    print(f"\nStrategy 2 - Clustered with hard negatives:")
    print(f"  Average similarities to GT: {np.mean(sims_2):.3f} ± {np.std(sims_2):.3f}")
    print(f"  Average similarities to hard negatives: {np.mean(hard_neg_sims_2):.3f} ± {np.std(hard_neg_sims_2):.3f}")
    
    # Strategy 3: Hierarchical
    gt_vecs_3 = generator.strategy_3_hierarchical_structure(query)
    avg_3 = np.mean(gt_vecs_3, axis=0)
    sims_3 = [np.dot(avg_3, gt) for gt in gt_vecs_3]
    print(f"\nStrategy 3 - Hierarchical:")
    print(f"  Average similarities to GT: {np.mean(sims_3):.3f} ± {np.std(sims_3):.3f}")
    print(f"  Min similarity: {np.min(sims_3):.3f}")
    
    # Strategy 4: Anti-averaging
    gt_vecs_4 = generator.strategy_4_anti_averaging_structure(query)
    avg_4 = np.mean(gt_vecs_4, axis=0)
    sims_4 = [np.dot(avg_4, gt) for gt in gt_vecs_4]
    print(f"\nStrategy 4 - Anti-averaging:")
    print(f"  Average similarities to GT: {np.mean(sims_4):.3f} ± {np.std(sims_4):.3f}")
    print(f"  Min similarity: {np.min(sims_4):.3f}")
    
    print(f"\n=== Recommendation ===")
    print(f"Strategy 4 (Anti-averaging) seems most challenging for single-vector retrieval!")
    print(f"Strategy 2 (Hard negatives) adds confounding factors in the corpus.")
    print(f"Combining strategies 2 + 4 would create the hardest benchmark.")


if __name__ == "__main__":
    test_strategies() 