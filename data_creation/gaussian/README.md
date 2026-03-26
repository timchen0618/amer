# Synthetic Gaussian Data Generation for Information Retrieval
This directory contains procedure for generating synthetic datasets to evaluate retrieval models. The generated datasets consist of d-dimensional vectors with known ground truth relationships.

## Overview

The synthetic data generation follows this process:

1. **Query Generation**: Sample query vectors from multivariate Gaussian N(0, I)
2. **Tranformation Matrices**: Create K matrices that are sufficiently different. The current method for generating transformation matrices is to generate two rotation matrices M_a and M_b. The five matrices are Identity matrix I, M_a, -M_a, M_b, -M_b. 
3. **Ground Truth Generation**: Transform queries using transformation matrices: y_i = A_i × x
4. **Corpus Construction**: Combine ground truth vectors with random vectors to form a searchable corpus.

### Transformation Type
There are two transformation types, "linear" and "MLP". "Linear" data is implemented using rotation matrices, and "MLP" data is implemented using two-layer MLPs (multi-layer perceptron). The source for the actual implementation can be found in `src_data_generation/`.

### Query Distribution Type
There are three query distribution types, including `single-in-distribution`, `multi-in-distribution`, and `ood` (Out-of-distribution). `single-in-distribution` is implemented with a single Gaussian distribution. For `multi-in-distribution`, we implement five different query distributions, and the model is trained on tested on the same distributions. For `ood`, we use the same five distribution as in `multi-in-distribution`, except we train on the first four and evaluate on the last distribution. More details can be found in the paper. The source for the actual implementation can be found in `src_data_generation/`.

## Files

#### Data Generation Scripts
- `src_data_generation/generate_linear.py` - Main data generation script for hard "linear" data (for single retriever). 
- `src_data_generation/generate_mlps.py`   - Main data generation script for hard "MLP" data. 

#### Loading / Testing / Verification
- `baseline_evaluation.py` - Test the orcale single retriever baseline. 
- `test.py` - for testing the model trained using the main training script. (In main folder)


## Usage
To generate all the synthetic data, simply run the following command to generate the synthetic data: 
```
bash create_syn_data.sh
```


### Basic Data Generation

Generate data with default parameters (1024-dimensional vectors, 2000 training + 200 test queries):

```bash
PYTHONPATH=. python src_data_generation/generate_linear.py --dimensions 1024 --train-queries 2000 --test-queries 200 --corpus-size 100000 --seed 42 --output-dir ./data/linear
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

Each query has K ground truth vectors in the corpus (one for each transformation).

