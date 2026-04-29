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

## INF-Retriever Fine-tuning Pipeline

This is a self-contained sub-pipeline under `training/inf_retriever/` that fine-tunes [`infly/inf-retriever-v1-1.5b`](https://huggingface.co/infly/inf-retriever-v1-1.5b) for multi-query retrieval. It is independent of the main `train.py` / `src/model.py` stack.

The full workflow has three stages:

```
1. Fine-tune the retriever  →  checkpoints/<run_name>/checkpoint/best_model/
2. Embed the corpus         →  wikipedia_embeddings/<mode>/<shard_id>   (pickle)
3. Run retrieval inference  →  results/<output_file>.jsonl
```

---

### Stage 1 — Fine-tuning

**Entry points**

| Script | Purpose |
|---|---|
| `training/inf_retriever/finetuning_multi.py` | Main training script |
| `training/inf_retriever/finetune.sbatch` | SLURM job (2× H200, `accelerate launch`) |
| `training/inf_retriever/finetune.sh` | Local multi-GPU launch (same args) |

**What `finetuning_multi.py` does, step by step**

1. Parses args via `Options` (`training/inf_retriever/src/options.py`).
2. Instantiates the model:
   - `training_mode=standard_org_q` → `EmbeddingModelDocEncNoProjSingleQuery` (single embedding per query, `ContrastiveLoss`)
   - `training_mode=multi` → `EmbeddingModelDocEncNoProj` (autoregressive multi-embedding, `HungarianMaskedContrastiveLoss` by default)
   Both classes live in `training/inf_retriever/src/inbatch.py` and load the INF-Retriever base weights via `transformers.AutoModel`.
3. Builds optimizer (AdamW) and LR scheduler (linear warmup) via `set_optim` in `training/inf_retriever/src/utils.py`.
4. Loads training/eval data via `prepare_data()`:
   - Reads JSONL with `Dataset` or `SampleDataset` (`training/inf_retriever/src/finetuning_data.py`).
   - In `multi` mode, uses `GoldLengthGroupedBatchSampler` to ensure each batch has uniform gold count (= uniform `k`).
   - Tokenizes and left-pads queries with `CollatorMulti` or `CollatorDocEncMultiQuery`.
5. Wraps everything with `Accelerator` (bf16, W&B logging under project `amer`).
6. Training loop: every `eval_freq` steps calls `evaluate()` (in-file), which reloads the model from the state dict and measures accuracy/MRR on the eval set.
7. Saves checkpoint via `save_state_dict` (`training/inf_retriever/src/utils.py`) only when MRR improves. Checkpoint format: `{step, model (state dict), optimizer, scheduler, opt}` → `checkpoints/<run_name>/checkpoint/best_model/checkpoint.pth`.

**Supporting modules (training)**

| File | Role |
|---|---|
| `training/inf_retriever/src/inbatch.py` | `EmbeddingModelDocEncNoProj`, `EmbeddingModelDocEncNoProjSingleQuery`, `InBatch`; loss functions: `ContrastiveLoss`, `HungarianContrastiveLoss`, `HungarianMaskedContrastiveLoss`; `build_loss()` factory |
| `training/inf_retriever/src/finetuning_data.py` | `Dataset`, `SampleDataset` (JSONL loaders); `GoldLengthGroupedBatchSampler`; collators `CollatorMulti`, `CollatorDocEncMultiQuery` |
| `training/inf_retriever/src/options.py` | All CLI arguments (`Options` class) |
| `training/inf_retriever/src/utils.py` | `set_optim` (AdamW + scheduler), `save_state_dict` / `load`, `WeightedAvgStats` (running loss tracker), `WarmupLinearScheduler`, `CosineScheduler` |
| `training/inf_retriever/src/dist_utils.py` | Distributed helpers: `gather` (with gradients), `varsize_gather` (variable-size all_gather with gradients, used by loss functions), `get_varsize`, `get_rank`, `get_world_size`, `weighted_average` |
| `training/inf_retriever/src/inf_retriever.py` | `INFRetriever` model wrapper (last-token pooling) |

---

### Stage 2 — Corpus Embedding Generation

Before retrieval, every passage in the corpus must be embedded and saved as sharded pickle files.

**Entry points**

| Script | Purpose |
|---|---|
| `gen_embed_new.py` | Embeds one shard of the passage corpus |
| `gen_embed.sbatch` | SLURM array job — launches one job per shard (e.g. `--array=0-31` for 32 shards) |

**What `gen_embed_new.py` does, step by step**

1. Detects whether `--model_name_or_path` is a fine-tuned checkpoint (contains `checkpoint.pth`) or a HuggingFace model.
   - Fine-tuned: loads via `load_retriever` (`src/inference_utils.py`), which reads `opt.training_mode` from the checkpoint to pick the correct model class, then extracts the underlying encoder (`model.encoder` for multi, `model` for standard).
   - Otherwise: loads as `SentenceTransformer` (inf-retriever, stella, NV-Embed) or `AutoModel`.
2. Reads its assigned shard of the TSV corpus (`--shard_id` / `--num_shards`).
3. Embeds passages with last-token pooling + L2 normalization, in batches of `--per_gpu_batch_size`.
4. Saves `(ids, embeddings)` as a pickle file to `--output_dir/<prefix>_<shard_id:02d>`.

**Supporting modules (embedding)**

| File | Role |
|---|---|
| `src/inference_utils.py` | `load_retriever` (loads fine-tuned checkpoint, selects model class from `opt.training_mode`); `embed_queries_stella`, `embed_queries`, `embed_queries_iterative_retrieval` (query embedding helpers for different model types) |

---

### Stage 3 — Retrieval Inference

**Entry points**

| Script | Purpose |
|---|---|
| `retrieval_inf.py` | Main inference script: embed queries → FAISS search → write output JSONL |
| `scripts/run_retrieval.sh` | Convenience wrapper with pre-filled paths |
| `retrieval_base.py` | Provides `retrieve()` (sharded FAISS search + aggregation) and `load_data()` — imported by `retrieval_inf.py` |

**What `retrieval_inf.py` does, step by step**

1. Detects checkpoint type (same logic as `gen_embed_new.py`).
   - Fine-tuned: loads via `load_retriever` (`src/inference_utils.py`); uses `embed_queries_single` (last-token pool with instruct format).
   - Otherwise: dispatches to `embed_queries_stella`, `embed_queries`, or `embed_queries_iterative_retrieval`.
2. Loads the passage corpus (TSV) via `load_passages` (`src/retrieval_utils.py`) into an id→passage map.
3. Embeds all queries into a numpy array, L2-normalized.
4. Calls `retrieve()` from `retrieval_base.py`, which:
   - Iterates over `--num_shards` shards of the pre-built embeddings.
   - For each shard, calls `load_index()` (`src/retrieval_utils.py`) to build a FAISS `Indexer` from the pickle files, then runs `search_knn`.
   - Aggregates per-shard results via `aggregate_sharded_results` (`src/retrieval_utils.py`), merging and re-sorting by score.
5. Attaches retrieved passages to each query example via `add_passages` (`src/retrieval_utils.py`).
6. Writes output JSONL — each line is the input example with a `ctxs` field (list of `{id, title, text, score}`).

**Supporting modules (retrieval)**

| File | Role |
|---|---|
| `src/retrieval_utils.py` | `Indexer` (FAISS flat/IVF-PQ index, `search_knn`, serialize/deserialize); `load_index` (builds index from shard pickles, optional save/load); `index_encoded_data` (streams pickles into index); `shard_and_get_embedding_files` (assigns pickle files to a shard); `aggregate_sharded_results` (merges and re-ranks across shards); `add_passages` (attaches retrieved docs to query examples); `load_passages` (reads TSV or JSONL corpus) |
| `src/inference_utils.py` | `load_retriever`, query embedding helpers (see Stage 2) |
| `retrieval_base.py` | `retrieve()` (orchestrates sharded FAISS search), `load_data()` |

---

### Data Format (INF-Retriever Pipeline)

**Training / eval JSONL** — each line:
```json
{
  "question_text": "...",
  "ground_truths": [{"id": "...", "title": "...", "text": "..."}, ...],
  "negative_ctxs":  [{"id": "...", "title": "...", "text": "..."}, ...],
  "hard_negative_ctxs": [{"id": "...", "title": "...", "text": "..."}, ...]
}
```
Eval data uses `positive_ctxs` instead of `ground_truths`.

**Corpus TSV** — tab-separated, columns: `id`, `text`, `title` (header row included).

**Passage embeddings** — one pickle file per shard, containing `(ids: list[str], embeddings: np.ndarray[N, 1536])`.

**Retrieval output JSONL** — each line is the input example augmented with:
```json
"ctxs": [{"id": "...", "title": "...", "text": "...", "score": "..."}, ...]
```

---

### Checkpoint Format

Saved by `save_state_dict` in `training/inf_retriever/src/utils.py`:
```
checkpoints/<run_name>/checkpoint/best_model/checkpoint.pth
```
The `.pth` file is a dict with keys: `step`, `model` (state dict), `optimizer`, `scheduler`, `opt` (the full `Options` namespace, including `training_mode` and `retriever_model_id`).

`load_retriever` (`src/inference_utils.py`) reads `opt.training_mode` from this saved `opt` to reconstruct the correct model class at inference time.

## Dependencies

Key packages: `torch`, `transformers`, `peft` (LoRA), `datasets`, `faiss`, `sentence-transformers`, `accelerate`, `wandb`/`trackio`.
