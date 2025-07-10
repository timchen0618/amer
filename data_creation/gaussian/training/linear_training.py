#!/usr/bin/env python3
"""
Linear Training Script for Synthetic Information Retrieval Dataset

This script trains a simple linear classifier to transform query vectors 
to ground truth vectors using the opposing pairs synthetic dataset.
"""

import numpy as np
import json
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any
import random
from tqdm import tqdm
import wandb


def load_synthetic_dataset(data_dir='./data/opposing_pairs_data/', split='train'):
    """
    Load the synthetic dataset.
    
    Args:
        data_dir: Path to the directory containing the synthetic data files
        split: 'train', 'test', or 'all'
        
    Returns:
        Tuple of (pairs_data, queries, corpus)
    """
    
    # Load configuration
    with open(os.path.join(data_dir, 'config.json'), 'r') as f:
        config = json.load(f)
    
    # Load data arrays
    print('loading corpus')
    corpus = np.load(os.path.join('data/opposing_pairs_data', 'corpus.npy'))
    print('loading queries')
    queries = np.load(os.path.join(data_dir, 'queries.npy'))
    
    # Load query-ground truth mappings
    with open(os.path.join(data_dir, 'query_ground_truth_pairs.json'), 'r') as f:
        pairs_data = json.load(f)
    
    if split == 'all':
        return pairs_data, queries, corpus
    else:
        return pairs_data[split], queries, corpus


def create_train_dev_split(train_pairs, dev_ratio=0.2, random_seed=42):
    """
    Split training data into train and development sets.
    
    Args:
        train_pairs: List of training pairs
        dev_ratio: Ratio of data to use for development set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_split, dev_split)
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle the pairs
    shuffled_pairs = train_pairs.copy()
    random.shuffle(shuffled_pairs)
    
    # Split into train and dev
    dev_size = int(len(shuffled_pairs) * dev_ratio)
    dev_split = shuffled_pairs[:dev_size]
    train_split = shuffled_pairs[dev_size:]
    
    return train_split, dev_split


class QueryGroundTruthDataset(Dataset):
    """
    Dataset for query-ground truth pairs.
    """
    
    def __init__(self, pairs_data, queries, corpus):
        """
        Args:
            pairs_data: List of query-ground truth pair dictionaries
            queries: Array of all query vectors
            corpus: Array of all corpus vectors
            augment_factor: How many times to augment each query (since each query has multiple GTs)
        """
        self.pairs_data = pairs_data
        self.queries = queries
        self.corpus = corpus
        
        # Create training pairs: for each query, randomly select one of its ground truth vectors
        self.training_pairs = []
        for pair in pairs_data:
            query_idx = pair['query_idx']
            query_vector = queries[query_idx]
            # gt_idx = pair['ground_truth_indices'][0]            
            # gt_vector = corpus[gt_idx]
            gt_vector = query_vector
            self.training_pairs.append((query_vector, gt_vector))
    
    def __len__(self):
        return len(self.training_pairs)
    
    def __getitem__(self, idx):
        query_vector, gt_vector = self.training_pairs[idx]
        return torch.FloatTensor(query_vector), torch.FloatTensor(gt_vector)


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


class MLPTransformer(nn.Module):
    """
    Multi-layer perceptron transformation model.
    """
    
    def __init__(self, input_dim, output_dim=None, hidden_dims=[512, 256], 
                 activation='relu', dropout=0.1, use_batch_norm=True):
        super(MLPTransformer, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        
        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'linear':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
                
            layers.append(self.activation)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
                
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)


def train_model(model, train_dataloader, dev_dataloader=None, queries=None, corpus=None, 
                num_epochs=100, learning_rate=0.001, device='cpu', k_values=[10, 20, 50, 100]):
    """
    Train the linear model with optional development set evaluation.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_epochs)
    
    best_dev_loss = float('inf')
    best_epoch = 0
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Training phase
        model.train()
        for batch_queries, batch_gts in tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs} (Train)'):
            batch_queries = batch_queries.to(device)
            batch_gts = batch_gts.to(device)
            # Forward pass
            outputs = model(batch_queries)
            loss = criterion(outputs, batch_gts)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Log training metrics
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": avg_loss
        }
        
        # Development evaluation
        with torch.no_grad():
            if dev_dataloader is not None and queries is not None and corpus is not None:
                # dev_results = evaluate_model_on_dataloader(
                #     model, dev_dataloader, queries, corpus, k_values, device
                # )
                
                # # Add dev metrics to log
                # for metric, value in dev_results.items():
                #     log_dict[f"dev_{metric}"] = value
                
                # # Track best model
                # current_recall = dev_results.get('Recall@10', 0.0)
                # if current_recall > best_dev_recall:
                #     best_dev_recall = current_recall
                #     best_epoch = epoch + 1
                #     log_dict["best_dev_recall"] = best_dev_recall
                #     log_dict["best_epoch"] = best_epoch
                
                
                dev_losses = []
                for batch_queries, batch_gts in dev_dataloader:
                    batch_queries = batch_queries.to(device)
                    batch_gts = batch_gts.to(device)
                    outputs = model(batch_queries)
                    loss = criterion(outputs, batch_gts)
                    dev_losses.append(loss.item())
                avg_dev_loss = sum(dev_losses) / len(dev_losses)
                log_dict["dev_loss"] = avg_dev_loss
                
                current_loss = avg_dev_loss
                if current_loss < best_dev_loss:
                    best_dev_loss = current_loss
                    best_epoch = epoch + 1
                    log_dict["best_dev_loss"] = best_dev_loss
                    log_dict["best_epoch"] = best_epoch
        
        # Log to wandb
        wandb.log(log_dict)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.6f}')
            if dev_dataloader is not None:
                # print(f'  Dev Recall@10: {dev_results.get("Recall@10", 0.0):.4f} (Best: {best_dev_recall:.4f} at epoch {best_epoch})')
                print(f'  Dev Loss: {avg_dev_loss:.6f} (Best: {best_dev_loss:.6f} at epoch {best_epoch})')
    return model


def evaluate_model_on_dataloader(model, dataloader, queries, corpus, k_values, device):
    """
    Evaluate model on a specific dataloader (for dev set evaluation during training).
    """
    model.eval()
    
    test_predictions = []
    test_gt_indices = []
    
    with torch.no_grad():
        for batch_queries, batch_gts in dataloader:
            batch_queries = batch_queries.to(device)
            
            # Get model predictions
            predictions = model(batch_queries).cpu().numpy()
            
            # For this evaluation, we need to get the ground truth indices
            # Since we only have the GT vectors in the dataloader, we need to find their indices
            batch_gts_np = batch_gts.numpy()
            
            for i, prediction in enumerate(predictions):
                test_predictions.append(prediction)
                
                # Find the index of the ground truth vector in the corpus
                gt_vector = batch_gts_np[i]
                # Find closest match in corpus (should be exact)
                similarities = np.dot(corpus, gt_vector) / (np.linalg.norm(corpus, axis=1) * np.linalg.norm(gt_vector) + 1e-8)
                gt_idx = np.argmax(similarities)
                test_gt_indices.append([gt_idx])  # Wrap in list for compatibility
    
    test_predictions = np.array(test_predictions)
    
    # Compute retrieval metrics
    results = compute_retrieval_metrics(
        test_predictions, corpus, test_gt_indices, k_values
    )
    
    return results


def evaluate_model(model, test_pairs, queries, corpus, k_values=[10, 20, 50, 100], device='cpu'):
    """
    Evaluate the model on retrieval performance.
    
    Args:
        model: Trained model
        test_pairs: Test query-ground truth pairs
        queries: All query vectors
        corpus: All corpus vectors
        k_values: List of k values for Recall@k computation
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    print("Evaluating model...")
    
    # Generate predictions for all test queries
    test_predictions = []
    test_query_indices = []
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
            test_query_indices.append(query_idx)
            test_gt_indices.append(gt_indices)
    
    test_predictions = np.array(test_predictions)
    
    # Compute retrieval metrics
    results = compute_retrieval_metrics(
        test_predictions, corpus, test_gt_indices, k_values
    )
    
    return results


def compute_retrieval_metrics(predictions, corpus, ground_truth_indices_list, k_values):
    """
    Compute Recall@k and MRecall@k metrics.
    
    Args:
        predictions: Predicted vectors (n_queries, embedding_dim)
        corpus: Full corpus (n_corpus, embedding_dim) 
        ground_truth_indices_list: List of ground truth indices for each query
        k_values: List of k values to compute metrics for
        
    Returns:
        Dictionary with metrics
    """
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


def main():
    parser = argparse.ArgumentParser(description='Train linear classifier for synthetic dataset')
    parser.add_argument('--data-dir', type=str, default='./data/opposing_pairs_data_large/', 
                       help='Directory containing synthetic data')
    parser.add_argument('--batch-size', '-b', type=int, default=32, 
                       help='Batch size for training')
    parser.add_argument('--num-epochs', '-e', type=int, default=100, 
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--k-values', nargs='+', type=int, default=[10, 20, 50, 100, 200, 500],
                       help='K values for Recall@k evaluation')
    parser.add_argument('--save-model', type=str, default='linear_model.pth',
                       help='Path to save trained model')
    parser.add_argument('--wandb-project', type=str, default='synthetic_gaussian',
                       help='Wandb project name')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--dev-ratio', type=float, default=0.2,
                       help='Ratio of training data to use for development set')
    parser.add_argument('--dev-eval-freq', type=int, default=1,
                       help='Evaluate on dev set every N epochs')
    
    # Model architecture arguments
    parser.add_argument('--model-type', type=str, default='linear', choices=['linear', 'mlp'],
                       help='Type of model to train (linear or mlp)')
    parser.add_argument('--hidden-dims', nargs='+', type=int, default=[512, 256],
                       help='Hidden dimensions for MLP (e.g., --hidden-dims 512 256 128)')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu', 'tanh', 'linear'],
                       help='Activation function for MLP')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate for MLP')
    parser.add_argument('--use-batch-norm', action='store_true',
                       help='Use batch normalization in MLP')
    
    args = parser.parse_args()
    
    # Update model save path based on model type
    if args.save_model == 'linear_model.pth':  # Only update if using default
        if args.model_type == 'mlp':
            hidden_str = '_'.join(map(str, args.hidden_dims))
            args.save_model = f'mlp_model_{hidden_str}_{args.activation}.pth'
    
    # Initialize wandb
    if not args.no_wandb:
        run_name = f"{args.model_type}_bs{args.batch_size}_lr{args.learning_rate}_ep{args.num_epochs}"
        if args.model_type == 'mlp':
            hidden_str = '_'.join(map(str, args.hidden_dims))
            run_name += f"_h{hidden_str}_{args.activation}"
            if args.dropout > 0:
                run_name += f"_d{args.dropout}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "model_type": args.model_type,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "k_values": args.k_values,
                "data_dir": args.data_dir,
                "dev_ratio": args.dev_ratio,
                "dev_eval_freq": args.dev_eval_freq,
                "hidden_dims": args.hidden_dims if args.model_type == 'mlp' else None,
                "activation": args.activation if args.model_type == 'mlp' else None,
                "dropout": args.dropout if args.model_type == 'mlp' else None,
                "use_batch_norm": args.use_batch_norm if args.model_type == 'mlp' else None,
            },
            tags=[f"{args.model_type}_training"]
        )
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading training data...")
    train_pairs, queries, corpus = load_synthetic_dataset(args.data_dir, split='train')
    print("Loading test data...")
    test_pairs, test_queries, test_corpus = load_synthetic_dataset(args.data_dir, split='test')
    
    # Create train/dev split
    print(f"Creating train/dev split with ratio {args.dev_ratio}...")
    train_split, dev_split = create_train_dev_split(train_pairs, args.dev_ratio)
    
    print(f"Dataset info:")
    print(f"  Original training pairs: {len(train_pairs)}")
    print(f"  Train split: {len(train_split)}")
    print(f"  Dev split: {len(dev_split)}")
    print(f"  Test pairs: {len(test_pairs)}")
    print(f"  Query dimension: {queries.shape[1]}")
    print(f"  Corpus size: {corpus.shape[0]}")
    
    # Log dataset info to wandb
    if not args.no_wandb:
        wandb.config.update({
            "original_train_pairs": len(train_pairs),
            "train_pairs": len(train_split),
            "dev_pairs": len(dev_split),
            "test_pairs": len(test_pairs),
            "query_dimension": queries.shape[1],
            "corpus_size": corpus.shape[0],
        })
    
    # Create datasets and dataloaders
    train_dataset = QueryGroundTruthDataset(train_split, queries, corpus)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    dev_dataset = QueryGroundTruthDataset(dev_split, queries, corpus)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")
    
    # Initialize model
    input_dim = queries.shape[1]
    
    if args.model_type == 'linear':
        model = LinearTransformer(input_dim)
        print(f"Created Linear model with input dim: {input_dim}")
    elif args.model_type == 'mlp':
        model = MLPTransformer(
            input_dim=input_dim,
            hidden_dims=args.hidden_dims,
            activation=args.activation,
            dropout=args.dropout,
            use_batch_norm=args.use_batch_norm
        )
        print(f"Created MLP model with architecture: {input_dim} -> {' -> '.join(map(str, args.hidden_dims))} -> {input_dim}")
        print(f"  Activation: {args.activation}")
        print(f"  Dropout: {args.dropout}")
        print(f"  Batch norm: {args.use_batch_norm}")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Log model info to wandb
    if not args.no_wandb:
        wandb.config.update({
            "model_parameters": num_params,
            "input_dim": input_dim,
            "training_samples": len(train_dataset),
            "dev_samples": len(dev_dataset)
        })
    
    # Train model with development evaluation
    print(f"\nStarting {args.model_type.upper()} training with development evaluation...")
    model = train_model(
        model, train_dataloader, dev_dataloader, queries, corpus,
        num_epochs=args.num_epochs, 
        learning_rate=args.learning_rate,
        device=device,
        k_values=args.k_values[:4]  # Use first 4 k-values for faster dev evaluation
    )
    
    # Save model
    torch.save(model.state_dict(), args.save_model)
    print(f"Model saved to {args.save_model}")
    
    # # Evaluate model
    # print("\nEvaluating model...")
    # results = evaluate_model(model, test_pairs, queries, corpus, args.k_values, device)
    
    # # Print results
    # print("\n" + "="*50)
    # print("EVALUATION RESULTS")
    # print("="*50)
    # for metric, value in results.items():
    #     print(f"{metric}: {value:.4f}")
    
    # # Also evaluate baseline (using query directly)
    # print("\nEvaluating Query Baseline...")
    # query_predictions = []
    # baseline_gt_indices = []
    
    # for pair in test_pairs:
    #     query_idx = pair['query_idx']
    #     query_vector = queries[query_idx]
    #     gt_indices = pair['ground_truth_indices']
        
    #     query_predictions.append(query_vector)
    #     baseline_gt_indices.append(gt_indices)
    
    # query_predictions = np.array(query_predictions)
    # baseline_results = compute_retrieval_metrics(
    #     query_predictions, corpus, baseline_gt_indices, args.k_values
    # )
    
    # print("\n" + "="*50)
    # print("BASELINE RESULTS (Query as Prediction)")
    # print("="*50)
    # for metric, value in baseline_results.items():
    #     print(f"{metric}: {value:.4f}")
    
    # # Print comparison
    # print("\n" + "="*50)
    # print("IMPROVEMENT OVER BASELINE")
    # print("="*50)
    # for metric in results.keys():
    #     improvement = results[metric] - baseline_results[metric]
    #     print(f"{metric}: {improvement:+.4f}")
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main() 