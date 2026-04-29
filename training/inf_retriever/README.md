# INF-Retriever Fine-tuning

Fine-tunes [`infly/inf-retriever-v1-1.5b`](https://huggingface.co/infly/inf-retriever-v1-1.5b) to produce **multiple query embeddings per question** via an autoregressive generation loop, enabling retrieval of documents covering diverse answer aspects.

## Overview

The core model (`EmbeddingModelDocEncNoProj`) runs the retriever encoder autoregressively: it encodes the query, extracts the last-token hidden state as the first query embedding, appends that hidden state back as an input token, runs the encoder again to produce a second embedding, and repeats for `k` steps. This yields `k` distinct query vectors from a single question, each targeting a different relevant document.

Two training modes are supported:

| Mode | Model class | Description |
|---|---|---|
| `standard_org_q` | `EmbeddingModelDocEncNoProjSingleQuery` | Single embedding per query. Uses `ContrastiveLoss`. Standard in-batch negatives. |
| `multi` | `EmbeddingModelDocEncNoProj` | Multiple embeddings per query (`k = number of gold documents`). Uses `hungarian_masked` loss by default. |

## Running Training

### On a SLURM cluster (recommended)

Edit hyperparameters at the top of `finetune.sbatch`, then:

```bash
sbatch training/inf_retriever/finetune.sbatch
```

The job requests 2 H200 GPUs and launches via `accelerate launch`.

### Locally (multi-GPU)

Edit `finetune.sh` as needed, then run from the repo root:

```bash
bash training/inf_retriever/finetune.sh
```

Both scripts call `finetuning_multi.py` via `accelerate launch`.

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--train_data` | — | Path(s) to training JSONL file(s) |
| `--eval_data` | — | Path(s) to evaluation JSONL file(s) |
| `--training_mode` | `standard_org_q` | `standard_org_q` or `multi` |
| `--loss_fn` | `auto` | `auto`, `contrastive`, `hungarian_masked`, `hungarian` |
| `--temperature` | `1.0` | InfoNCE temperature |
| `--total_steps` | `1000` | Total training steps |
| `--warmup_steps` | `-1` | Linear warmup steps |
| `--lr` | `1e-4` | Learning rate |
| `--per_gpu_batch_size` | `64` | Batch size per GPU |
| `--negative_ctxs` | `1` | Number of negatives per positive |
| `--negative_hard_ratio` | `0.0` | Fraction of negatives that are hard negatives |
| `--chunk_length` | `256` | Max token length for passages |
| `--accumulation_steps` | `1` | Gradient accumulation steps |
| `--norm_query` / `--norm_doc` | off | L2-normalize query/document embeddings |
| `--eval_freq` | `500` | Evaluate every N steps |
| `--save_freq` | `50000` | Save checkpoint every N steps (only when eval improves) |
| `--output_dir` | `./checkpoint/` | Directory to save checkpoints |
| `--run_name` | `my_experiments` | Run name (used for W&B logging and checkpoint subdirectory) |

`--loss_fn auto` resolves to `hungarian_masked` when `training_mode=multi`, and `contrastive` otherwise.

## Data Format

Training and evaluation data are JSONL files. Each line is a JSON object with:

```json
{
  "question_text": "...",
  "ground_truths": [
    {"id": "...", "title": "...", "text": "..."},
    ...
  ],
  "negative_ctxs": [{"id": "...", "title": "...", "text": "..."}, ...],
  "hard_negative_ctxs": [{"id": "...", "title": "...", "text": "..."}, ...]
}
```

Evaluation data uses `positive_ctxs` instead of `ground_truths`.

In `multi` mode, each example can have multiple ground truths. The `GoldLengthGroupedBatchSampler` groups examples by gold count so each batch contains examples with the same number of positives, keeping `k` uniform within a batch.

## Loss Functions

### `contrastive` (`ContrastiveLoss`)
Standard InfoNCE loss. Each of the `k` predicted embeddings is matched to its corresponding gold document at the same position, using in-batch negatives gathered across all ranks.

### `hungarian_masked` (`HungarianMaskedContrastiveLoss`) — default for `multi`
Extends contrastive loss with two improvements:
1. **Hungarian assignment**: uses the scipy linear-sum-assignment algorithm to find the optimal matching between the `k` predicted embeddings and the `k` gold documents, removing the "embedding j must match gold j" constraint.
2. **Positive masking**: when computing the softmax denominator, masks out the other `k-1` same-example positives so they don't act as false negatives. Only the Hungarian-assigned positive and all negatives from other examples remain in the loss denominator.

Logs both strict accuracy (argmax == Hungarian target) and relaxed accuracy (argmax falls within any same-example positive).

### `hungarian` (`HungarianContrastiveLoss`)
Hungarian matching without positive masking. Kept for comparison; the same-example false-negative issue means the loss is inflated when the model correctly ranks other positives.

## Model Architecture

`EmbeddingModelDocEncNoProj` wraps the INF-Retriever encoder with:
- **Gradient checkpointing** enabled to reduce memory usage
- **Last-token pooling** following the INF-Retriever documentation
- **Scheduled sampling**: at each autoregressive step (for `multi` mode), the predicted hidden state is fed back with probability `sampling_rate = step / total_steps`, ramping from teacher-forced to self-generated over training

The document encoder is the **same shared encoder** used for queries — documents are encoded independently (no projection from/to a different space).

## Evaluation During Training

Every `eval_freq` steps, `evaluate()` runs on the eval set and reports:
- `eval_acc`: fraction of queries where the top-1 retrieved doc is a gold doc
- `eval_mrr`: mean reciprocal rank

The best checkpoint (by MRR) is saved to `{output_dir}/{run_name}/best_model/`.

Metrics are logged to W&B under the project `amer`.

## Inference

Retrieval with a fine-tuned checkpoint is handled by `retrieval_inf.py` (repo root), using pre-built corpus embeddings and a FAISS index.

### Running

Edit the paths at the top of `scripts/run_retrieval.sh`, then run from the repo root:

```bash
bash scripts/run_retrieval.sh
```

Or call the script directly:

```bash
python retrieval_inf.py \
    --model_name_or_path checkpoints/<run_name>/checkpoint/best_model/ \
    --passages /path/to/chunks.tsv \
    --passages_embeddings "wikipedia_embeddings/standard/*" \
    --data data/amer_data/eval_data/qampari.jsonl \
    --output_dir results/ \
    --output_file qampari.jsonl \
    --projection_size 1536 \
    --n_docs 500 \
    --num_shards 16 \
    --use_gpu
```

### Model Loading

`retrieval_inf.py` auto-detects the checkpoint type by looking for `checkpoint.pth` inside the given path:

- **Fine-tuned checkpoint** (contains `checkpoint.pth`): loaded via `load_retriever` from `src/inference_utils.py`; queries are embedded using last-token pooling with the INF-Retriever instruct format.
- **HuggingFace / SentenceTransformer path** (no `checkpoint.pth`): loaded as a `SentenceTransformer` or `AutoModel`. Supported: `inf-retriever`, `stella`, `NV-Embed`, `iterative_retrieval`.

### Key Inference Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name_or_path` | required | Checkpoint dir or HF model ID |
| `--passages` | — | Path to corpus TSV (`id`, `text`, `title`) |
| `--passages_embeddings` | — | Glob pattern for pre-built passage embedding shards |
| `--data` | required | Query JSONL (fields: `question` or `question_text`) |
| `--output_dir` | — | Directory for the output JSONL |
| `--output_file` | `output.jsonl` | Output filename |
| `--n_docs` | `100` | Number of documents to retrieve per query |
| `--projection_size` | `1536` | Embedding dimension (INF-Retriever hidden size) |
| `--num_shards` | `8` | Number of passage embedding shards |
| `--per_gpu_batch_size` | `16` | Query encoding batch size |
| `--use_gpu` | off | Use GPU for FAISS indexing/search |
| `--no_fp16` | off | Disable fp16 (model runs in fp32) |
| `--save_embeddings` | off | Save query embeddings to disk and exit (skip retrieval) |

### Output Format

Each line of the output JSONL is the input example augmented with a `ctxs` field:

```json
{
  "question": "...",
  "ctxs": [
    {"id": "...", "title": "...", "text": "...", "score": "..."},
    ...
  ]
}
```

Results are ranked by descending retrieval score.

## Source Files

| File | Purpose |
|---|---|
| `finetuning_multi.py` | Main training script: data loading, training loop, evaluation |
| `finetune.sbatch` | SLURM job script for 2-GPU H200 training |
| `finetune.sh` | Local multi-GPU launch script |
| `src/inbatch.py` | Model classes (`EmbeddingModelDocEncNoProj`, `InBatch`) and loss functions |
| `src/finetuning_data.py` | Dataset classes, collators, `GoldLengthGroupedBatchSampler` |
| `src/options.py` | All CLI arguments |
| `src/inf_retriever.py` | INF-Retriever model wrapper |
| `src/dist_utils.py` | Distributed training utilities (gather, varsize gather) |
| `src/utils.py` | Optimizer/scheduler setup, logging |
