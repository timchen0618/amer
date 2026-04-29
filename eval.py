import argparse
import csv
import pandas as pd
from src.eval_utils import read_jsonl, eval_retrieve_docs, evaluate, mrr
import numpy as np


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate retrieval performance')
    
    # Data configuration
    parser.add_argument('--data_path', type=str,
                       default='amer_data/eval_data/qampari_dev_data_gt_qampari_corpus_5_to_8_ctxs.jsonl',
                       help='Data path')
    
    # Evaluation settings
    parser.add_argument('--topk', type=int, nargs='+', default=[99, 10],
                       help='Top-k values for evaluation')
    
    # File configuration
    parser.add_argument('--input-file', type=str, default=None,
                       help='Custom list of files to evaluate')
    parser.add_argument('--selected-indices-file', type=str, default=None,
                       help='Path to file containing selected indices')
    
    # Other options
    parser.add_argument('--no-gold-id', action='store_true',
                       help='Whether the data has gold IDs')
    
    return parser.parse_args()


def load_selected_indices(file_path):
    """Load selected indices from file."""
    if file_path is None:
        return None
    try:
        with open(file_path, 'r') as f:
            return [int(line.strip()) for line in f]
    except FileNotFoundError:
        print(f"Warning: Selected indices file {file_path} not found. Using all indices.")
        return None


def main():
    args = parse_arguments()
    
    # Load selected indices if provided
    selected_indices = load_selected_indices(args.selected_indices_file)
    print("selected_indices", selected_indices)

    for topk in args.topk:
        print('-' * 40)
        print(f"Evaluating: {args.input_file}")
        print(f"Top-k: {topk}")
        print(f"Data path: {args.data_path}")
        print("Has gold ID: ", not args.no_gold_id)
        
        # Main evaluation
        eval_retrieve_docs(
            args.input_file,
            args.data_path,
            has_gold_id=not args.no_gold_id,
            topk=topk,
            selected_indices=selected_indices
        )

if __name__ == "__main__":
    main()