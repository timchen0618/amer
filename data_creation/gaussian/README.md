# Synthetic Gaussian Data Generation for Information Retrieval

This directory contains tools for generating synthetic datasets to evaluate information retrieval models. The generated datasets consist of d-dimensional vectors with known ground truth relationships.

## Overview

The synthetic data generation follows this process:

1. **Query Generation**: Sample query vectors from multivariate Gaussian N(0, I)
2. **Tranformation Matrices**: Create K matrices that are sufficiently different. The current method for generating transformation matrices is to generate two rotation matrices M_a and M_b. The five matrices are Identity matrix I, M_a, -M_a, M_b, -M_b. 
3. **Ground Truth Generation**: Transform queries using transformation matrices: y_i = A_i × x
4. **Corpus Construction**: Combine ground truth vectors with random vectors to form a searchable corpus.

## Files

#### Data Generation Scripts
- `generate_data.py` - Main data generation script, without being difficult to a single retriever model. 
- `generate_data_opposing_pairs.py` - Main data generation script for hard data (for single retriever). Implement the transformation matrices idea in [Overview](#overview).
- `hard_dataset_ideas.py` - some ideas for generating hard data. 

#### Loading / Testing / Verification
- `baseline_evaluation.py` - Test the orcale single retriever baseline. 
- `loading_instructions.py` - Example of how to load and evaluate with the generated data
- `test_anti_averaging.py` - Small-scale test to verify the generation works
- `validate_data.py` - validate if the data is as what we thought. 
- `test.py` - for testing the model trained using the main training script (`autoregressive/`)


## Usage

### Basic Data Generation

Generate data with default parameters (1024-dimensional vectors, 2000 training + 200 test queries):

```bash
PYTHONPATH=. python src_data_generation/generate_data_opposing_pairs.py --dimensions 1024 --train-queries 2000 --test-queries 200 --corpus-size 100000 --seed 42 --output-dir ./data/opposing_pairs_data
```

### Custom Parameters

Generate data with custom parameters:

```bash
python generate_data.py \
    --dimensions 1024 \
    --train-queries 20000 \
    --test-queries 1000 \
    --corpus-size 250000 \
    --output-dir ./data/opposing_pairs_data_large
```

### Command Line Options

- `--dimensions, -d`: Dimensionality of vectors (default: 128)
- `--train-queries`: Number of training queries (default: 2000)
- `--test-queries`: Number of test queries (default: 200)
- `--corpus-size`: Total corpus size (default: 100000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir, -o`: Output directory (default: ./synthetic_data)

## Generated Files

The script generates the following files:

- `corpus.npy` - The searchable corpus (corpus_size × d array)
- `queries.npy` - All query vectors (n_queries × d array)
- `transformation_matrices.npy` - The K transformation matrices (K × d × d array)
- `query_ground_truth_pairs.json` - Query-ground truth mappings for train/test splits
- `config.json` - Configuration parameters used for generation

## Data Structure

### Query-Ground Truth Pairs Format

```json
{
  "train": [
    {
      "query_idx": 0,
      "ground_truth_indices": [1245, 5672, 8901, ...]
    },
    ...
  ],
  "test": [...],
  "metadata": {
    "n_train_queries": 2000,
    "n_test_queries": 200,
    "n_transformations": 5,
    "dimensions": 1024,
    "corpus_size": 100000
  }
}
```

Each query has K ground truth vectors in the corpus (one for each transformation matrix).

