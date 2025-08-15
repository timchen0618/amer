import argparse
import pandas as pd
from src.eval_utils import read_jsonl, eval_retrieve_docs, eval_retrieve_docs_for_repeats, evaluate, mrr

# root = '/scratch/cluster/hungting/projects/Multi_Answer/contriever/outputs/contriever_msmarco_nq/'
# root = '/scratch/cluster/hungting/projects/Multi_Answer/mteb_retriever/outputs/'

def get_data_mapping(project_root):
    """Returns the data mapping configuration."""
    return {
        "qampari": {
            'dev': f'{project_root}/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl',
            'dev_5_to_8': f'{project_root}/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus_5_to_8_ctxs.jsonl',
            'train': f'{project_root}/diverse_response/data/qampari_data/train_data_gt_qampari_corpus.jsonl',
            'second_stage': f'{project_root}/diverse_response/data/qampari_data/2nd_stage_test_data/dev_data_qampari_corpus_inp{{num_input}}.jsonl'
        },
        "ambiguous": {
            'dev': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_no_empty_clusters.jsonl',
            'dev_2_to_5': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.jsonl',
            'train': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data',
        },
        "ambiguous_qe": {
            'dev': f'{project_root}/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev.jsonl',
            'dev_2_to_5': f'{project_root}/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.jsonl',
            'train': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data',
        },
        "wsd_distinct": {
            'dev': f'{project_root}/autoregressive/data/wsd/distinct/dev.jsonl',
        },
    }


def get_file_list(data_type):
    """Returns the default file list for a given data type."""
    return [
        f'retrieval_out_dev_{data_type}_single.jsonl', 
        f'retrieval_out_dev_{data_type}.jsonl', 
        f'retrieval_out_dev_{data_type}_from_2nd_to_3rd.jsonl'
    ]
    # return [f'retrieval_out_dev_{data_type}.jsonl']

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate retrieval performance')
    
    # Main configuration
    parser.add_argument('--root', type=str, 
                       default='/scratch/hc3337/projects/autoregressive/results/nq_inf/',
                       help='Root directory for results')
    parser.add_argument('--project-root', type=str, 
                       default='/scratch/hc3337/projects/',
                       help='Project root directory')
    parser.add_argument('--reranking-root', type=str,
                       default='/scratch/cluster/hungting/projects/diverse_response/retrieval_outputs/qampari_2nd_stage/',
                       help='Reranking root directory')
    
    # Data configuration
    parser.add_argument('--data-type', type=str, default='wsd_distinct',
                       choices=['qampari', 'ambiguous', 'ambiguous_qe', 'wsd_distinct'],
                       help='Type of data to evaluate')
    parser.add_argument('--split', type=str, default=None,
                       help='Data split to use (will be auto-determined if not provided)')
    parser.add_argument('--num-input', type=int, nargs='+', default=[3],
                       help='Number of inputs to evaluate')
    
    # Evaluation settings
    parser.add_argument('--topk', type=int, nargs='+', default=[99, 10],
                       help='Top-k values for evaluation')
    parser.add_argument('--reranking', action='store_true',
                       help='Enable reranking evaluation')
    parser.add_argument('--second-stage', action='store_true',
                       help='Enable second stage evaluation')
    
    # File configuration
    parser.add_argument('--file-list', type=str, nargs='+', default=None,
                       help='Custom list of files to evaluate')
    parser.add_argument('--selected-indices-file', type=str, default=None,
                       help='Path to file containing selected indices')
    parser.add_argument('--output-csv', type=str, default='evaluation_scores.csv',
                       help='Output CSV file name')
    
    # Other options
    parser.add_argument('--has-gold-id', action='store_true',
                       help='Whether the data has gold IDs')
    
    return parser.parse_args()


def get_split_for_data_type(data_type):
    """Get default split for data type if not specified."""
    if data_type == 'qampari':
        return 'dev_5_to_8'
    elif data_type in ['ambiguous', 'ambiguous_qe']:
        return 'dev_2_to_5'
    elif data_type == 'wsd_distinct':
        return 'dev'
    else:
        return 'dev'


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
    
    # Set up configuration
    data_mapping = get_data_mapping(args.project_root)
    all_scores = []
    
    # Determine split if not provided
    if args.split is None:
        args.split = get_split_for_data_type(args.data_type)
    
    # Load selected indices if provided
    selected_indices = load_selected_indices(args.selected_indices_file)
    print("selected_indices", selected_indices)
    
    # Determine file list
    if args.file_list is None:
        file_list = get_file_list(args.data_type)
    else:
        file_list = args.file_list
    
    print(f"Evaluating data type: {args.data_type}")
    print(f"Split: {args.split}")
    print(f"Number of inputs: {args.num_input}")
    print(f"Top-k values: {args.topk}")
    print(f"Files to evaluate: {file_list}")
    print(f"Root: {args.root}")
    print("-" * 50)
    
    for num_input in args.num_input:
        # Get data path
        if args.data_type == 'qampari' and args.second_stage:
            data_path = data_mapping[args.data_type]['second_stage'].format(num_input=num_input)
        else:
            data_path = data_mapping[args.data_type][args.split]
        
        # Set root directory
        eval_root = args.reranking_root if args.reranking else args.root
        
        for file_name in file_list:
            scores_per_file = []
            mrecalls = []
            recalls = []
            
            for topk in args.topk:
                scores_per_file.append(0)  # Placeholder
                input_file = eval_root + file_name
                
                print('-' * 40)
                print(f"Evaluating: {file_name}")
                print(f"Top-k: {topk}")
                print(f"Data path: {data_path}")
                
                # Main evaluation
                scores = eval_retrieve_docs(
                    input_file,
                    data_path,
                    has_gold_id=args.has_gold_id,
                    topk=topk,
                    selected_indices=selected_indices
                )
                
                qrels = scores[-2]
                runs = scores[-1]
                main_scores = scores[:-2]
                
                # Second stage evaluation if enabled
                if args.second_stage:
                    repeat_scores = eval_retrieve_docs_for_repeats(
                        input_file,
                        data_path,
                        topk=topk
                    )
                    all_eval_scores = list(main_scores) + list(repeat_scores)
                else:
                    all_eval_scores = list(main_scores)
                
                scores_per_file.extend(all_eval_scores)
                print(f"Number of scores: {len(all_eval_scores)}")
                
                if len(all_eval_scores) > 0:
                    mrecalls.append(all_eval_scores[0])
                if len(all_eval_scores) > 1:
                    recalls.append(all_eval_scores[1])
            
            # Collect scores for this file
            if len(args.topk) >= 2:
                # Assuming we have scores for both topk values
                score_indices = [1, 2, 3, 5, 8, 9, 10, 12] if len(scores_per_file) > 12 else [1,2,3,6,7,8]
                file_scores = [scores_per_file[i] if i < len(scores_per_file) else 0 for i in score_indices]
                all_scores.append(file_scores)
    
    # Save results to CSV
    if args.second_stage:
        columns = ['MRecall', 'Recall', 'Precision', 'mAP', 'nDCG', 'MRR', 
                  'MRecall-Repeat', 'Recall-Repeat', 'Precision-Repeat', '-', 
                  'MRecall', 'Recall', 'Precision', 'mAP', 'nDCG', 'MRR', 
                  'MRecall-Repeat', 'Recall-Repeat', 'Precision-Repeat']
    else:
        columns = ['MRecall@100', 'Recall@100', 'Precision@100', 'nDCG@100', 
                  'MRecall@10', 'Recall@10', 'Precision@10', 'nDCG@10']
    
    df = pd.DataFrame(all_scores, columns=columns[:len(all_scores[0]) if all_scores else len(columns)], index=file_list)
    df.index.name = 'file_name'
    df.to_csv(args.output_csv, index=True)
    print(f"\nResults saved to: {args.output_csv}")
    print(f"Shape: {df.shape}")
    print("\nSample results:")
    print(df.head())


if __name__ == "__main__":
    main()