# Argument Parser for gen_ret_and_eval.py

This document describes the modifications made to `gen_ret_and_eval.py` to add comprehensive command-line argument support.

## Changes Made

### 1. Added Argument Parser
- Added `argparse` import
- Created `parse_args()` function with comprehensive argument definitions
- Replaced all hardcoded variables with argument references

### 2. Removed Config File Dependency
- Removed dependency on `configs/eval.yaml` file
- Converted all config parameters to command-line arguments
- Updated `eval_with_generation()` function to use arguments directly instead of config objects

### 3. Configurable Parameters

#### Data Configuration
- `--data_name`: Dataset to evaluate on (default: 'ambiguous_qe')
- `--training_data_name`: Dataset used for training (default: 'ambiguous_qe')
- `--split`: Data split to evaluate on (default: 'dev')
- `--suffix_list`: List of model suffixes to evaluate (default: ["_contrastive_from_stage2_lr2e5_ep20_temp0.05_warmup0.05"])
- `--retriever_list`: List of retrievers to use (default: ['inf'])

#### Model Configuration
- `--base_model_id`: Base model ID (default: "meta-llama/Llama-3.2-1B-Instruct")
- `--max_new_tokens`: Maximum number of new tokens to generate (default: None)
- `--compute_loss`: Whether to compute loss during evaluation (default: True)

#### Config Parameters (previously from config file)
- `--loss_function`: Loss function to use (default: 'Contrastive')
- `--question_only`: Whether to use question only (default: False)
- `--batch_size_training`: Batch size for training (default: 1)
- `--use_gt_q_embed`: Whether to use ground truth question embedding (default: True)
- `--use_eos`: Whether to use end of sequence token (default: False)

#### Paths and Directories
- `--embeddings_root`: Root directory for embeddings (default: '/scratch/hc3337/embeddings/')
- `--root`: Root directory for data (default: '/scratch/hc3337')

#### Retrieval Configuration
- `--top_k_per_query`: Number of top results per query (default: 100)
- `--top_k`: Number of top results to aggregate (default: 100)
- `--start_idx`: Starting index for data processing (default: 0)
- `--end_idx`: Ending index for data processing (default: None)
- `--aggregate_start_idx`: Starting index for aggregation (default: 0)
- `--aggregate_end_idx`: Ending index for aggregation (default: None)

#### Indexing Configuration
- `--use_gpu`: Whether to use GPU for indexing (default: False)
- `--num_shards`: Number of shards for indexing (default: 1)
- `--save_or_load_index`: Whether to save/load index (default: False)

#### Google API Configuration
- `--google_api`: Whether to use Google API (default: False)

#### Checkpoint Mapping
- `--checkpoint_num`: Checkpoint number to use (default: 1501)
- `--use_suffix_mapping`: Whether to use suffix-based checkpoint mapping (default: True)

## Usage Examples

### Basic Usage
```bash
python gen_ret_and_eval.py
```

### Different Dataset and Retriever
```bash
python gen_ret_and_eval.py \
    --data_name nq \
    --training_data_name nq \
    --retriever_list stella inf \
    --suffix_list "_contrastive_lr2e5_ep20_temp0.05_warmup0.05"
```

### Google API Usage
```bash
python gen_ret_and_eval.py \
    --data_name arguana_generated \
    --google_api \
    --retriever_list inf
```

### Custom Paths and Configuration
```bash
python gen_ret_and_eval.py \
    --data_name msmarco \
    --embeddings_root /custom/embeddings/path \
    --root /custom/data/path \
    --base_model_id meta-llama/Llama-3.2-3B-Instruct \
    --max_new_tokens 5 \
    --top_k 50 \
    --top_k_per_query 200 \
    --use_gpu \
    --num_shards 4
```

### Custom Checkpoint Number
```bash
python gen_ret_and_eval.py \
    --data_name ambiguous_qe \
    --checkpoint_num 2500 \
    --use_suffix_mapping false
```

### Custom Config Parameters
```bash
python gen_ret_and_eval.py \
    --data_name ambiguous_qe \
    --loss_function MSE \
    --question_only \
    --batch_size_training 4 \
    --use_gt_q_embed false \
    --use_eos
```

### Help Message
```bash
python gen_ret_and_eval.py --help
```

## Backward Compatibility

The script maintains backward compatibility by:
1. Using sensible defaults that match the original hardcoded values
2. Preserving all existing functionality
3. Maintaining the same output structure and file naming conventions

## Validation

The script includes validation for:
- Google API usage requirements for specific datasets
- Valid dataset and retriever choices
- Compatible argument combinations

## Testing

Run the example usage script to test different argument combinations:
```bash
python example_usage.py
```

This will demonstrate various usage patterns and validate that the argument parser works correctly.

## Benefits

1. **Flexibility**: Easy to experiment with different configurations without code changes
2. **Reproducibility**: Command-line arguments can be saved in scripts or documentation
3. **Automation**: Easy to integrate into automated workflows and experiments
4. **Documentation**: Self-documenting through help messages and argument descriptions
5. **Maintainability**: Centralized configuration management 