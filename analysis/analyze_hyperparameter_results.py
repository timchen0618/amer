#!/usr/bin/env python3
"""
Hyperparameter Search Results Analysis Script

This script analyzes the results from hyperparameter search experiments
and provides summary statistics and rankings.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

def parse_experiment_name(exp_name: str) -> Dict[str, float]:
    """Parse experiment name to extract hyperparameters."""
    pattern = r'lr([0-9e\-\.]+)_temp([0-9e\-\.]+)_batch(\d+)_ep(\d+)_warmup([0-9e\-\.]+)'
    match = re.match(pattern, exp_name)
    
    if not match:
        return {}
    
    return {
        'learning_rate': float(match.group(1)),
        'temperature': float(match.group(2)),
        'batch_size': int(match.group(3)),
        'num_epochs': int(match.group(4)),
        'warmup_ratio': float(match.group(5))
    }

def extract_metrics_from_log(log_file: str) -> Dict[str, float]:
    """Extract training metrics from log file."""
    metrics = {
        'final_train_loss': None,
        'best_val_loss': None,
        'total_steps': None,
        'converged': False
    }
    
    if not os.path.exists(log_file):
        return metrics
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # Extract final training loss
        train_loss_pattern = r'completed \(loss\): ([0-9\.]+)'
        train_losses = re.findall(train_loss_pattern, content)
        if train_losses:
            metrics['final_train_loss'] = float(train_losses[-1])
            
        # Extract validation loss
        val_loss_pattern = r'eval loss.*?([0-9\.]+)'
        val_losses = re.findall(val_loss_pattern, content)
        if val_losses:
            metrics['best_val_loss'] = min(float(loss) for loss in val_losses)
            
        # Check if training completed
        if 'saving model' in content.lower() or 'training completed' in content.lower():
            metrics['converged'] = True
            
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return metrics

def collect_results(results_dir: str, outputs_dir: str) -> pd.DataFrame:
    """Collect results from all experiments."""
    results = []
    
    # Look for experiment directories
    for exp_dir in Path(results_dir).iterdir():
        if not exp_dir.is_dir():
            continue
            
        exp_name = exp_dir.name
        hyperparams = parse_experiment_name(exp_name)
        
        if not hyperparams:
            continue
            
        # Look for corresponding log file
        log_file = Path(outputs_dir) / f"run_{exp_name}.out"
        metrics = extract_metrics_from_log(str(log_file))
        
        # Combine hyperparameters and metrics
        result = {**hyperparams, **metrics, 'experiment_name': exp_name}
        results.append(result)
    
    return pd.DataFrame(results)

def analyze_results(df: pd.DataFrame, metric: str = 'best_val_loss') -> None:
    """Analyze and visualize hyperparameter search results."""
    
    if df.empty:
        print("No results found!")
        return
    
    print(f"\n=== Hyperparameter Search Results Analysis ===")
    print(f"Total experiments: {len(df)}")
    print(f"Completed experiments: {df['converged'].sum()}")
    print(f"Failed experiments: {(~df['converged']).sum()}")
    
    # Filter to completed experiments
    completed_df = df[df['converged']].copy()
    
    if completed_df.empty:
        print("No completed experiments found!")
        return
    
    print(f"\nAnalyzing {len(completed_df)} completed experiments...")
    
    # Sort by metric (ascending for loss, descending for accuracy)
    ascending = 'loss' in metric.lower()
    completed_df = completed_df.sort_values(metric, ascending=ascending)
    
    # Top 10 results
    print(f"\n=== Top 10 Results (by {metric}) ===")
    top_10 = completed_df.head(10)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        print(f"{i:2d}. {row['experiment_name']}")
        print(f"    {metric}: {row[metric]:.6f}")
        print(f"    LR: {row['learning_rate']}, Temp: {row['temperature']}, "
              f"Batch: {row['batch_size']}, Epochs: {row['num_epochs']}, "
              f"Warmup: {row['warmup_ratio']}")
        print()
    
    # Best hyperparameters analysis
    print(f"\n=== Best Hyperparameter Analysis ===")
    hyperparams = ['learning_rate', 'temperature', 'batch_size', 'num_epochs', 'warmup_ratio']
    
    for param in hyperparams:
        best_value = completed_df.iloc[0][param]
        param_stats = completed_df.groupby(param)[metric].agg(['mean', 'std', 'count'])
        print(f"\n{param}:")
        print(f"  Best value: {best_value}")
        print(f"  Statistics by {param}:")
        print(param_stats.round(6))
    
    # Correlation analysis
    print(f"\n=== Correlation with {metric} ===")
    correlations = completed_df[hyperparams + [metric]].corr()[metric].drop(metric)
    correlations = correlations.sort_values(key=abs, ascending=False)
    for param, corr in correlations.items():
        print(f"  {param:15s}: {corr:+.4f}")

def create_visualizations(df: pd.DataFrame, metric: str = 'best_val_loss', output_dir: str = 'plots') -> None:
    """Create visualizations of hyperparameter search results."""
    
    completed_df = df[df['converged']].copy()
    if completed_df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    hyperparams = ['learning_rate', 'temperature', 'batch_size', 'num_epochs', 'warmup_ratio']
    
    # 1. Distribution plots for each hyperparameter
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, param in enumerate(hyperparams):
        if i < len(axes):
            sns.boxplot(data=completed_df, x=param, y=metric, ax=axes[i])
            axes[i].set_title(f'{metric} vs {param}')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Remove empty subplot
    if len(hyperparams) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hyperparameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation heatmap
    correlation_matrix = completed_df[hyperparams + [metric]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Hyperparameter Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Learning rate vs temperature heatmap
    pivot_table = completed_df.pivot_table(values=metric, index='learning_rate', columns='temperature', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.4f')
    plt.title(f'{metric} by Learning Rate and Temperature')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/lr_temp_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Analyze hyperparameter search results")
    parser.add_argument("--results_dir", type=str, default="results/hypersearch", 
                       help="Directory containing experiment results")
    parser.add_argument("--outputs_dir", type=str, default="sbatch_outputs",
                       help="Directory containing SBATCH output logs")
    parser.add_argument("--metric", type=str, default="best_val_loss",
                       help="Metric to analyze (best_val_loss, final_train_loss)")
    parser.add_argument("--output_csv", type=str, default="hyperparameter_results.csv",
                       help="Output CSV file for results")
    parser.add_argument("--plot", action="store_true", help="Create visualization plots")
    parser.add_argument("--plot_dir", type=str, default="plots", 
                       help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Collect results
    print("Collecting results from experiments...")
    df = collect_results(args.results_dir, args.outputs_dir)
    
    if df.empty:
        print("No experiments found!")
        return
    
    # Save results to CSV
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    
    # Analyze results
    analyze_results(df, args.metric)
    
    # Create visualizations if requested
    if args.plot:
        try:
            create_visualizations(df, args.metric, args.plot_dir)
        except ImportError:
            print("\nVisualization requires matplotlib and seaborn. Install with:")
            print("pip install matplotlib seaborn")

if __name__ == "__main__":
    main() 