# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research implementation for **multi-query retrieval using autoregressive LMs**. The model fine-tunes language models (Llama-1B/3B/8B, Qwen-3-4B) to auto-regressively generate *multiple* query embeddings per question, enabling retrieval of documents covering diverse aspects of a query. See paper: "Beyond Single Embeddings: Capturing Diverse Targets with Multi-Query Retrieval".

## Commands

### Training

```bash
# Single-GPU training (configure args directly)
python train.py --project <wandb_project> --save_path <output_dir> --train_path <data_dir> \
    --model_id meta-llama/Llama-3.2-1B-Instruct \
    --loss_function Hungarian_Contrastive --temperature 0.05

# Multi-GPU training (edit scripts/single_run.sh first, then run)
bash scripts/single_run.sh
```

The shell script auto-detects GPU count and calls `train_distributed.py` (via accelerate) for multi-GPU.

### Data Creation

```bash
# Full pipeline (edit corpus path in script first)
bash data_creation/create_data.sh

# Individual steps:
python data_creation/process_chunks.py        # TSV corpus from Wikipedia chunks
python data_creation/generate_embeddings.py   # Embed corpus with retriever
python data_creation/create_input_dataset.py  # Create HF training datasets
```

### Retrieval & Evaluation (Real Data)

```bash
# Generate embeddings + retrieve + evaluate
python gen_ret_and_eval.py \
    --data_name ambiguous_qe \
    --base_model_id meta-llama/Llama-3.2-1B-Instruct \
    --adapter_path <checkpoint_dir> \
    --linear_checkpoint_path <linear_ckpt> \
    --top_k 100 --max_new_tokens 5

# Or use the retrieval script
bash scripts/run_retrieval.sh

# Evaluate saved retrieval results
python eval.py --data_path amer_data/eval_data/qampari.jsonl \
    --input-file <retrieval_output> --topk 100 50 10
```

### Evaluation (Synthetic/Gaussian Data)

```bash
bash scripts/eval/test_gaussian.sh
# Or directly:
python test.py --model_paths <checkpoint_dir> \
    --raw_data_dir data_creation/gaussian/data/linear \
    --embedding_data_dir synthetic_datasets/synthetic_linear_test/
```

## Architecture

### Core Model (`src/model.py`)

`EmbeddingModel` bridges two embedding spaces through three components:
1. `base_causallm` — base LM (Llama/Qwen), optionally LoRA fine-tuned
2. `input_projection` — maps retriever embeddings → LM hidden space (injects doc embeddings as tokens)
3. `output_projection` — maps LM outputs → retriever embedding space

**Auto-regressive generation loop** (`model.generate()`): each step takes the hidden state at the current position, projects it to retriever space to produce one embedding, appends it as the next "token" via `input_projection`, and repeats for `max_new_tokens` steps → yields k embeddings per query.

Loss functions: `ContrastiveLoss`, `HungarianContrastiveLoss`, `MSELoss`, `HungarianMSELoss`. Hungarian variants use the scipy assignment algorithm for optimal matching between predicted and target embeddings.

### Data Flow

Training data is pre-computed HuggingFace datasets on disk containing: `input_ids`, `attention_mask`, `positive_embeddings` (tensor), `negative_embeddings` (tensor). The collators in `src/dataset.py` handle left-padding and EOS token insertion.

Pipeline: raw JSONL questions → tokenize → embed corpus with external retriever (inf-retriever-v1) → store as tensors → `DataHandler` loads via `datasets.load_from_disk()`.

### Training Modes (controlled by `--mode` and `--schedule_sampling`)

- `single` — one embedding per query (standard retriever fine-tuning)
- `multi_scheduled_sampling` — curriculum: starts single-token, gradually increases `max_new_tokens`
- `multi_always_sampling` — always generate multiple embeddings

### Key Source Files

| File | Purpose |
|------|---------|
| `src/model.py` | `EmbeddingModel`, loss functions |
| `src/dataset.py` | `DataHandler`, collator classes |
| `src/option.py` | All ~50 CLI arguments with defaults |
| `src/retrieval_utils.py` | FAISS `Indexer` class |
| `src/eval_utils.py` | Recall/precision metrics, answer matching |
| `src/utils.py` | LR schedulers, optimizer setup |
| `gen_ret_and_eval.py` | Full eval pipeline (generate → retrieve → score) |
| `data_creation/create_input_dataset.py` | Build training datasets from embeddings |

### Important Arguments (`src/option.py`)

- `--loss_function`: `Contrastive`, `Hungarian_Contrastive`, `MSE`, `Hungarian_MSE`
- `--temperature`: InfoNCE temperature (default 0.05)
- `--max_new_tokens`: number of embeddings to generate per query
- `--full_finetuning`: full FT instead of LoRA (default is LoRA)
- `--normalize`: L2-normalize embeddings
- `--tie_weights`: tie input/output projection weights
- `--log_with`: `wandb` or `trackio`

## Data

- Training datasets: `training_datasets/<model>/<dataset>/<retriever>/`
- Eval data: `amer_data/eval_data/` (JSONL with questions + ground truth)
- Corpus: TSV with columns `id`, `text`, `title` (built from `chunked_wikipedia/`)
- Corpus embeddings: sharded numpy arrays in `output_embeddings/`

## Dependencies

Key packages: `torch`, `transformers`, `peft` (LoRA), `datasets`, `faiss`, `sentence-transformers`, `accelerate`, `wandb`/`trackio`.
