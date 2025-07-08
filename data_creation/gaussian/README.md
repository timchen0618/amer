# Synthetic Gaussian Data Generation for Information Retrieval

This directory contains tools for generating synthetic datasets to evaluate information retrieval models. The generated datasets consist of d-dimensional vectors with known ground truth relationships.

## Overview

The synthetic data generation follows this process:

1. **Query Generation**: Sample query vectors from multivariate Gaussian N(0, I)
2. **Rotation Matrices**: Create K orthogonal rotation matrices that are sufficiently different
3. **Ground Truth Generation**: Transform queries using rotation matrices: y_i = A_i × x
4. **Corpus Construction**: Combine ground truth vectors with random vectors to form a searchable corpus

## Files

#### Data Generation Scripts
- `generate_data.py` - Main data generation script
- `generate_data_opposing_pairs.py` - Main data generation script for hard data (for single retriever).
- `hard_dataset_ideas.py` - some ideas for generating hard data. 

#### Loading / Testing / Verification
- `baseline_evaluation.py` - Test the orcale single retriever baseline. 
- `loading_instructions.py` - Example of how to load and evaluate with the generated data
- `test_anti_averaging.py` - Small-scale test to verify the generation works
- `validate_data.py` - validate if the data is as what we thought. 

## Installation

Install the required dependencies:

```bash
pip install numpy scipy
# or
pip install -r requirements.txt
```

## Usage

### Basic Data Generation

Generate data with default parameters (128-dimensional vectors, 2000 training + 200 test queries):

```bash
python generate_data.py
```

### Custom Parameters

Generate data with custom parameters:

```bash
python generate_data.py \
    --dimensions 256 \
    --train-queries 5000 \
    --test-queries 500 \
    --rotations 10 \
    --corpus-size 200000 \
    --output-dir ./my_synthetic_data
```

### Command Line Options

- `--dimensions, -d`: Dimensionality of vectors (default: 128)
- `--train-queries`: Number of training queries (default: 2000)
- `--test-queries`: Number of test queries (default: 200)
- `--rotations, -k`: Number of rotation matrices (default: 5)
- `--corpus-size`: Total corpus size (default: 100000)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-dir, -o`: Output directory (default: ./synthetic_data)

## Generated Files

The script generates the following files:

- `corpus.npy` - The searchable corpus (corpus_size × d array)
- `queries.npy` - All query vectors (n_queries × d array)
- `rotation_matrices.npy` - The K rotation matrices (K × d × d array)
- `ground_truth_indices.npy` - Boolean array indicating which corpus vectors are ground truth
- `query_ground_truth_pairs.json` - Query-ground truth mappings for train/test splits
- `config.json` - Configuration parameters used for generation

## Data Structure

### Query-Ground Truth Pairs Format

```json
{
  "train": [
    {
      "query_idx": 0,
      "query_vector": [0.1, 0.2, ...],
      "ground_truth_indices": [1245, 5672, 8901, ...]
    },
    ...
  ],
  "test": [...],
  "metadata": {
    "n_train_queries": 2000,
    "n_test_queries": 200,
    "n_rotations": 5,
    "dimensions": 128,
    "corpus_size": 100000
  }
}
```

Each query has K ground truth vectors in the corpus (one for each rotation matrix).

## Example Usage

See `example_usage.py` for a complete example of loading and evaluating with the generated data:

```bash
python example_usage.py
```

This script demonstrates:
- Loading the generated data
- Computing similarity matrices
- Evaluating retrieval performance with Recall@K metrics
- Analyzing individual query results

## Testing

Run a small-scale test to verify the generation works:

```bash
python test_generation.py
```

## Key Features

1. **Configurable Parameters**: All dataset sizes and dimensions are easily adjustable
2. **Reproducible**: Uses fixed random seeds for consistent results
3. **Mathematically Sound**: Uses proper orthogonal matrices via QR decomposition
4. **Evaluation Ready**: Includes ground truth mappings and evaluation utilities
5. **Scalable**: Can generate datasets of arbitrary size (limited by memory)

## Theory

The generated data has known mathematical relationships:
- Each query x is sampled from N(0, I)
- Ground truth vectors y_i = A_i × x where A_i are orthogonal rotation matrices
- The cosine similarity between x and y_i depends on the rotation angle
- Random corpus vectors provide realistic noise for evaluation

This creates a controlled environment where the "correct" retrievals are known, allowing for precise evaluation of information retrieval models.

## Performance Notes

- Generation time scales with corpus size and dimensionality
- Memory usage is approximately: corpus_size × dimensions × 8 bytes for float64
- For large datasets (>1M vectors), consider generating in chunks if memory is limited 