# Linear Classifier Training for Synthetic Information Retrieval

This repository contains scripts to train and evaluate linear classifiers on synthetic information retrieval datasets. The goal is to learn a linear transformation that maps query vectors to ground truth vectors for improved retrieval performance.

## Overview

The synthetic dataset uses an "opposing pairs" structure where each query has 5 corresponding ground truth vectors created by applying 5 transformation matrices:
- 1 Identity matrix (I)
- 2 Linear transformations (A, B)
- 2 Negative transformations (-A, -B)

This creates an anti-averaging effect that makes simple averaging of ground truth vectors ineffective for retrieval.

## Files

- `linear_training.py` - Main training script
- `evaluate_linear_model.py` - Standalone evaluation script
- `demo.py` - End-to-end demo script
- `README_linear_training.md` - This documentation

## Quick Start

### Option 1: Run the demo script
```bash
python demo.py
```

This will:
1. Generate synthetic data (if not present)
2. Train a linear classifier
3. Evaluate the model and show results

### Option 2: Run individual steps

1. **Generate data** (if not already done):
```bash
python generate_data_opposing_pairs.py --dimensions 1024 --train-queries 2000 --test-queries 200 --corpus-size 100000 --output-dir ./data/opposing_pairs_data/
```

2. **Train the model**:
```bash
python linear_training.py --data-dir ./data/opposing_pairs_data/ --num-epochs 100 --batch-size 32 --learning-rate 0.001
```

3. **Evaluate the model**:
```bash
python evaluate_linear_model.py --model-path linear_model.pth --data-dir ./data/opposing_pairs_data/
```

## Training Script Details

### `linear_training.py`

**Purpose**: Trains a linear classifier to transform query vectors into ground truth vectors.

**Key Features**:
- Uses MSE loss to train the linear transformation
- Data augmentation: Each query is paired with randomly selected ground truth vectors
- Automatic baseline comparison (query-as-prediction)
- Comprehensive evaluation with Recall@k and MRecall@k metrics

**Arguments**:
- `--data-dir`: Directory containing synthetic data
- `--batch-size`: Training batch size (default: 32)
- `--num-epochs`: Number of training epochs (default: 100)
- `--learning-rate`: Learning rate (default: 0.001)
- `--augment-factor`: How many times to augment each query (default: 5)
- `--k-values`: K values for evaluation (default: [10, 20, 50, 100, 200, 500])
- `--save-model`: Path to save trained model (default: linear_model.pth)

**Example**:
```bash
python linear_training.py \
    --data-dir ./data/opposing_pairs_data/ \
    --batch-size 64 \
    --num-epochs 50 \
    --learning-rate 0.001 \
    --augment-factor 5
```

## Evaluation Script Details

### `evaluate_linear_model.py`

**Purpose**: Evaluates a trained linear model and compares it to baselines.

**Key Features**:
- Loads and evaluates any trained linear model
- Compares against two baselines:
  - Query Baseline: Uses query vector directly for retrieval
  - Average Baseline: Uses average of ground truth vectors
- Computes Recall@k and MRecall@k metrics

**Arguments**:
- `--model-path`: Path to trained model (required)
- `--data-dir`: Directory containing synthetic data
- `--split`: Data split to evaluate on ('train' or 'test')
- `--k-values`: K values for evaluation

**Example**:
```bash
python evaluate_linear_model.py \
    --model-path linear_model.pth \
    --data-dir ./data/opposing_pairs_data/ \
    --split test
```

## Model Architecture

The model is a simple linear transformation:
```python
class LinearTransformer(nn.Module):
    def __init__(self, input_dim, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
```

## Training Process

1. **Data Loading**: Each query can be paired with any of its 5 ground truth vectors
2. **Data Augmentation**: Each query is replicated multiple times with different GT vectors
3. **Loss Function**: MSE loss between predicted and target ground truth vectors
4. **Optimization**: Adam optimizer with configurable learning rate

## Evaluation Metrics

- **Recall@k**: Average fraction of relevant documents retrieved in top-k
- **MRecall@k**: Mean Recall - fraction of queries with at least one relevant document in top-k

For each query:
- Recall@k = (# relevant retrieved in top-k) / (# total relevant)
- MRecall@k = 1 if any relevant retrieved in top-k, else 0

## Expected Results

The linear classifier should outperform simple baselines:

1. **Query Baseline**: Using the query vector directly typically gives poor results due to the transformation structure
2. **Average Baseline**: Averaging ground truth vectors is suboptimal due to the opposing pairs structure (-A cancels +A)
3. **Linear Model**: Should learn to predict better retrieval vectors

## Dependencies

```bash
pip install torch numpy scipy scikit-learn tqdm
```

## Data Structure

The synthetic dataset contains:
- `corpus.npy`: All searchable vectors (ground truth + random)
- `queries.npy`: Query vectors (Gaussian distributed)
- `query_ground_truth_pairs.json`: Query-to-ground truth mappings
- `config.json`: Dataset configuration

Each query has 5 corresponding ground truth vectors in the corpus, making this a challenging multi-target retrieval task.

## Customization

You can modify the training by:
- Changing the model architecture in `LinearTransformer`
- Adjusting loss functions in the training loop
- Adding regularization or different optimizers
- Experimenting with different data augmentation strategies

## Troubleshooting

**Common Issues**:
1. **Data not found**: Make sure to generate the synthetic data first
2. **CUDA out of memory**: Reduce batch size or use CPU
3. **Poor performance**: Try different learning rates or more epochs

**Performance Tips**:
- Use GPU if available for faster training
- Increase batch size for stable gradients
- Monitor training loss to ensure convergence 