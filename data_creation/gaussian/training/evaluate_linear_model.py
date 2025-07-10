#!/usr/bin/env python3
"""
Evaluation Script for Trained Linear Models

This script loads a trained linear model and evaluates it on the synthetic dataset.
"""

import numpy as np
import json
import os
import argparse
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any


class LinearTransformer(nn.Module):
    """
    Simple linear transformation model.
    """
    
    def __init__(self, input_dim, output_dim=None):
        super(LinearTransformer, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)


def load_synthetic_dataset(data_dir='./data/opposing_pairs_data/', split='test'):
    """Load the synthetic dataset."""
    
    # Load data arrays
    corpus = np.load(os.path.join(data_dir, 'corpus.npy'))
    queries = np.load(os.path.join(data_dir, 'queries.npy'))
    
    # Load query-ground truth mappings
    with open(os.path.join(data_dir, 'query_ground_truth_pairs.json'), 'r') as f:
        pairs_data = json.load(f)
    
    return pairs_data[split], queries, corpus


def compute_retrieval_metrics(predictions, corpus, ground_truth_indices_list, k_values):
    """Compute Recall@k and MRecall@k metrics."""
    
    print("Computing retrieval metrics...")
    
    # Normalize vectors for cosine similarity
    predictions_norm = predictions / (np.linalg.norm(predictions, axis=1, keepdims=True) + 1e-8)
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities: (n_queries, n_corpus)
    similarities = predictions_norm @ corpus_norm.T
    
    results = {}
    
    for k in k_values:
        recalls = []
        mrecalls = []
        
        for i, gt_indices in enumerate(ground_truth_indices_list):
            # Get top-k most similar corpus vectors
            top_k_indices = np.argsort(similarities[i])[-k:][::-1]
            
            # Compute Recall@k
            num_relevant_retrieved = len(set(top_k_indices) & set(gt_indices))
            recall_k = num_relevant_retrieved / len(gt_indices)
            recalls.append(recall_k)
            
            # Compute MRecall@k (binary: 1 if any relevant retrieved, 0 otherwise)
            mrecall_k = 1.0 if num_relevant_retrieved > 0 else 0.0
            mrecalls.append(mrecall_k)
        
        # Average metrics
        results[f'Recall@{k}'] = np.mean(recalls)
        results[f'MRecall@{k}'] = np.mean(mrecalls)
    
    return results


def evaluate_model(model, test_pairs, queries, corpus, k_values=[10, 20, 50, 100], device='cpu'):
    """Evaluate the model on retrieval performance."""
    
    model.eval()
    model.to(device)
    
    print("Evaluating model...")
    
    # Generate predictions for all test queries
    test_predictions = []
    test_gt_indices = []
    
    with torch.no_grad():
        for pair in test_pairs:
            query_idx = pair['query_idx']
            query_vector = queries[query_idx]
            gt_indices = pair['ground_truth_indices']
            
            # Get model prediction
            query_tensor = torch.FloatTensor(query_vector).unsqueeze(0).to(device)
            prediction = model(query_tensor).cpu().numpy().squeeze()
            
            test_predictions.append(prediction)
            test_gt_indices.append(gt_indices)
    
    test_predictions = np.array(test_predictions)
    
    # Compute retrieval metrics
    results = compute_retrieval_metrics(
        test_predictions, corpus, test_gt_indices, k_values
    )
    
    return results


def evaluate_baselines(test_pairs, queries, corpus, k_values):
    """Evaluate baseline methods."""
    
    # Query baseline (use query directly)
    print("Evaluating Query Baseline...")
    query_predictions = []
    baseline_gt_indices = []
    
    for pair in test_pairs:
        query_idx = pair['query_idx']
        query_vector = queries[query_idx]
        gt_indices = pair['ground_truth_indices']
        
        query_predictions.append(query_vector)
        baseline_gt_indices.append(gt_indices)
    
    query_predictions = np.array(query_predictions)
    query_results = compute_retrieval_metrics(
        query_predictions, corpus, baseline_gt_indices, k_values
    )
    
    # Average baseline (average of all ground truth vectors for each query)
    print("Evaluating Average Baseline...")
    avg_predictions = []
    
    for pair in test_pairs:
        gt_indices = pair['ground_truth_indices']
        gt_vectors = corpus[gt_indices]
        avg_gt = np.mean(gt_vectors, axis=0)
        avg_predictions.append(avg_gt)
    
    avg_predictions = np.array(avg_predictions)
    avg_results = compute_retrieval_metrics(
        avg_predictions, corpus, baseline_gt_indices, k_values
    )
    
    return query_results, avg_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained linear model')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data-dir', type=str, default='./data/opposing_pairs_data/', 
                       help='Directory containing synthetic data')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                       help='Data split to evaluate on')
    parser.add_argument('--k-values', nargs='+', type=int, default=[10, 20, 50, 100, 200, 500],
                       help='K values for Recall@k evaluation')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading {args.split} data...")
    test_pairs, queries, corpus = load_synthetic_dataset(args.data_dir, split=args.split)
    
    print(f"Dataset info:")
    print(f"  Test pairs: {len(test_pairs)}")
    print(f"  Query dimension: {queries.shape[1]}")
    print(f"  Corpus size: {corpus.shape[0]}")
    
    # Load and evaluate trained model
    if os.path.exists(args.model_path):
        print(f"\nLoading model from {args.model_path}...")
        
        # Initialize model
        input_dim = queries.shape[1]
        model = LinearTransformer(input_dim)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # Evaluate model
        results = evaluate_model(model, test_pairs, queries, corpus, args.k_values, device)
        
        print("\n" + "="*50)
        print("TRAINED MODEL RESULTS")
        print("="*50)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
    else:
        print(f"Model file {args.model_path} not found!")
        results = None
    
    # Evaluate baselines
    print("\nEvaluating baselines...")
    query_results, avg_results = evaluate_baselines(test_pairs, queries, corpus, args.k_values)
    
    print("\n" + "="*50)
    print("QUERY BASELINE RESULTS")
    print("="*50)
    for metric, value in query_results.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n" + "="*50)
    print("AVERAGE BASELINE RESULTS")
    print("="*50)
    for metric, value in avg_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Print comparisons
    if results:
        print("\n" + "="*50)
        print("IMPROVEMENT OVER QUERY BASELINE")
        print("="*50)
        for metric in results.keys():
            improvement = results[metric] - query_results[metric]
            print(f"{metric}: {improvement:+.4f}")
        
        print("\n" + "="*50)
        print("IMPROVEMENT OVER AVERAGE BASELINE")
        print("="*50)
        for metric in results.keys():
            improvement = results[metric] - avg_results[metric]
            print(f"{metric}: {improvement:+.4f}")


if __name__ == '__main__':
    main() 