# Fallback recipe: joint doc-encoder multi-query training (pure bf16, 2-GPU DDP)

**Status:** the known-good but *buggy* recipe. It reproduces the April/May `with_detach` (QAMPARI) and
`multi_hungarian` (AmbigQA) results. Keep it as the fallback until a clean recipe beats it.
Verified 2026-09-23/24; git tag `fallback-bf16-ddp`.

> **Reproduce this only from the tag.** Since branch `fsdp-clean-recipe`, the training code
> always uses mixed precision (fp32 weights + fp32 AdamW, bf16 autocast), steps the LR scheduler
> once per step, and seeds a rank-consistent data order. It has no pure-bf16 mode, so the current
> code cannot reproduce this recipe. Use the tagged code, e.g.
> `git worktree add ../autoregressive-fallback fallback-bf16-ddp`, and run the scripts from there.
> Relative output paths (`checkpoints/`, `results/`, `sbatch_outputs/`) then land inside the worktree.

Full investigation (narrative, evidence, all tables):
https://claude.ai/code/artifact/ee8584b3-851e-4d5a-b7d4-a8f1a0c7e28e

| Dataset | Checkpoint evaluated | MRecall@100 | Recall@100 | MRecall@10 | Recall@10 |
|---|---|---|---|---|---|
| QAMPARI (k=5) | DDP rerun, step 250 | **26.37** | **58.62** | 1.32 | 21.99 |
| QAMPARI (k=5) | original April `with_detach`, step 250 | 25.42 | 56.90 | 1.69 | 21.79 |
| AmbigQA (k=2) | DDP rerun, step 750 | **75.09** | **87.43** | 45.83 | 67.90 |
| AmbigQA (k=2) | original May `multi_hungarian`, step 750 | 74.12 | 87.27 | 44.62 | 66.94 |

Run-to-run noise is about 1 point (data order is unseeded, see below).

---

## 1. What makes this recipe work (read before changing anything)

This recipe **depends on three bugs/quirks**. Fixing any one of them changes the result.

1. **Pure bf16 weights and optimizer (the one that matters most).**
   `_load_retriever` loads the model with `torch_dtype=torch.bfloat16`. Under plain DDP nothing upcasts
   it, so the weights *and* the AdamW moments stay bf16. Adam updates of about lr = 1e-5 fall below bf16
   resolution for most weights (half-ulp is about 2.4e-5 at the median |w| of 0.012), so they round to
   zero. Only about **7% (QAMPARI) / 10% (AmbigQA) of weights ever change**, and layernorms never do.
   - **FSDP breaks this.** With mixed precision on, accelerate upcasts FSDP FlatParameters to fp32
     (with or without `fsdp_offload_params`). Every weight then trains, and QAMPARI collapses to
     MRecall@100 = 0.00. AmbigQA degrades to 70–72.7, below the untrained base model (72.55).
   - **Never launch without `--config_file`.** The account default
     (`$HF_HOME/accelerate/default_config.yaml`) is FSDP + offload. In April, sbatch jobs happened not to
     pick it up (so they ran DDP) while interactive runs did. That's how the original results came to be
     bf16 by accident.
2. **LR scheduler steps twice per training step on 2 GPUs.** accelerate's `AcceleratedScheduler`
   calls `scheduler.step()` `num_processes` times per step (`split_batches=False`). The effective
   warmup is `warmup_steps/2`, and **LR reaches 0 at `total_steps/2`**:
   QAMPARI at step 2500 of 5000, AmbigQA at step 400 of 800. Everything after that is a no-op.
   This recipe runs on 2 processes, so it bakes the compression in.
3. **Unseeded Python `random` plus per-rank batch sharding.** Only `torch.manual_seed` is set.
   `GoldLengthGroupedBatchSampler` shuffles with Python `random`, independently on each rank, and
   accelerate's `BatchSamplerShard` gives each rank every other batch of *its own* permutation. Per
   epoch, about 25% of examples are skipped and about 25% are seen twice, and the two ranks usually
   hold different gold counts k at the same step. Runs aren't bit-reproducible.

Related quirks that are harmless here: `best_eval_metric` resets every epoch; the Hungarian cost-matrix
`.float()` is a no-op; `accelerate_config_2gpu_ddp.yaml` says `mixed_precision: fp16`, but the code's
hardcoded `Accelerator(mixed_precision="bf16")` overrides it (autocast only; params stay bf16).

## 2. Environment

| Item | Value |
|---|---|
| Hardware | 2× H200 (1 node), 128 GB RAM, 8 CPUs |
| Container | `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif` |
| Training overlay | `/scratch/hc3337/envs/div.ext3` (`source /ext3/env.sh`) |
| Retrieval overlay | `/scratch/hc3337/envs/nli.ext3` (`conda activate nli`) |
| Packages | torch 2.5.1+cu121, transformers 4.56.1, accelerate 1.9.0 |
| Accelerate config | `training/inf_retriever/accelerate_config_2gpu_ddp.yaml` (`distributed_type: MULTI_GPU`, `num_processes: 2`) |
| Code | git tag `fallback-bf16-ddp` |
| Base model | `infly/inf-retriever-v1-1.5b` |
| Model class | `EmbeddingModelDocEncNoProj` (joint doc encoder, `training_mode=multi`, with the teacher `.detach()`) |

## 3. Hyperparameters

Identical to the original runs' saved `opt`, apart from `--save_at_steps`, which only adds saves.

| Flag | QAMPARI | AmbigQA |
|---|---|---|
| `--train_data` / `--eval_data` | `data/training/filtered/qampari/{train,dev}_data.jsonl` | `data/training/filtered/ambigqa/{train,dev}_data.jsonl` |
| `--total_steps` | 5000 (LR = 0 from step 2500) | 800 (LR = 0 from step 400) |
| `--warmup_steps` | 200 (effective 100) | 30 (effective 15) |
| `--lr` | 1e-5 | 1e-5 |
| `--per_gpu_batch_size` | 50 (global 100) | 50 (global 100) |
| `--temperature` | 0.05 | 0.05 |
| `--loss_fn` / `--training_mode` | hungarian / multi | hungarian / multi |
| `--negative_ctxs` / `--negative_hard_ratio` | 1 / 0.0 | 1 / 0.0 |
| `--chunk_length` / `--max_positive_documents` | 512 / 1 | 512 / 1 |
| `--norm_query --norm_doc` | on | on |
| `--save_freq` / `--eval_freq` / `--log_freq` | 250 / 250 / 25 | 30 / 30 / 5 |
| Optimizer / schedule | AdamW (β 0.9/0.98, eps 1e-6, wd 0.01), linear warmup + decay | same |
| Seed | `--seed 0` (torch only) | same |
| **Checkpoint to evaluate** | **step 250** (`--save_at_steps 250 500`) | **step 750** (`--save_at_steps 390 750`) |
| Wall time | about 30 min to step 250 (+ step-0 eval) | about 45 min for all 800 steps |

## 4. How to run

```bash
cd /scratch/hc3337/projects/autoregressive

# 1. Train (2x H200, DDP)
sbatch training/inf_retriever/finetune_qampari_joint_repro_ddp.sh   # -> checkpoints/qampari/qampari_joint_repro_..._hungarian_ddp/checkpoint/step-250
sbatch training/inf_retriever/finetune_ambigqa_joint_repro_ddp.sh   # -> checkpoints/ambigqa/ambigqa_joint_repro_..._hungarian_ddp/checkpoint/step-750
# (the QAMPARI script runs to its walltime; only steps 250/500 are needed, so it can be cancelled after step 500)

# 2. Embed the corpus with THIS checkpoint (joint doc encoder -> must re-embed; 32-way array, ~75 GB)
sbatch gen_embed_qampari_ddp_step250.sbatch      # -> wikipedia_embeddings/qampari/ddp_step250/
sbatch gen_embed_ambigqa_ddp_step750.sbatch      # -> wikipedia_embeddings/ambigqa/ddp_step750/

# 3. Retrieve (k is required: QAMPARI 5, AmbigQA 2)
sbatch --dependency=afterok:<embed_jobid> retrieve_qampari_ddp_step250.sbatch   # -> results/finetuned/qampari/ddp_step250/qampari.jsonl
sbatch --dependency=afterok:<embed_jobid> retrieve_ambigqa_ddp_step750.sbatch   # -> results/finetuned/ambigqa/ddp_step750/ambigqa.jsonl

# 4. Evaluate (CPU is enough; run via srun, not on the login node)
python eval.py --data_path data/amer_data/eval_data/qampari.jsonl --topk 100 10 \
    --input-file results/finetuned/qampari/ddp_step250/qampari.jsonl
python eval.py --data_path data/amer_data/eval_data/ambigqa.jsonl --topk 100 10 --no-gold-id \
    --input-file results/finetuned/ambigqa/ddp_step750/ambigqa.jsonl
```

## 5. Fingerprints: how to tell a rerun is really this recipe

Check these before spending ~2 h and 75 GB on embedding + retrieval.

| Check | QAMPARI (expected) | AmbigQA (expected) |
|---|---|---|
| Log contains "Upcasted low precision parameters … FSDP" | **must be absent** | **must be absent** |
| Train loss | 4.69 @25, 3.74 @50, 1.21 @75, 0.59 @100, 0.58 @250 | 3.30 @5, 3.12 @10, 2.19 @15, 1.19 @20, 0.87 @30 |
| Logged `lr` | 2.5e-6 @25, 1e-5 @100, 9.38e-6 @250 | reaches 0 at step 400 |
| Training-time eval | step 250: pairwise_cos ≈ 0.697, repeat ≈ 77.4%; step 500: 0.781 | step 30: 0.771; flat at ≈ 0.752 from step ~390 |
| Checkpoint file | 9,260,049,422 B; bf16 weights; 338 full-shape bf16 AdamW states | same |
| Scheduler `last_epoch` in checkpoint | 2 × step (500 at step 250) | 2 × step (1500 at step 750) |
| Weights changed vs base | ≈ 7.0% at step 250 | ≈ 9.7% by the end |

An FSDP run looks clearly different: fp32 weights, 13.3 GB checkpoints, flattened optimizer state,
loss ≈ 0.67 at QAMPARI step 50, and ~100% of weights changed within 30 steps.

## 6. Artifacts to keep (do not delete)

| Artifact | Path |
|---|---|
| QAMPARI DDP checkpoint (step 250, evaluated) | `checkpoints/qampari/qampari_joint_repro_with_detach_steps5000_t0.05_lr0.00001_ws200_bs50_hungarian_ddp/checkpoint/step-250/` |
| AmbigQA DDP checkpoint (step 750, evaluated) | `checkpoints/ambigqa/ambigqa_joint_repro_steps800_t0.05_lr0.00001_ws30_bs50_hungarian_ddp/checkpoint/step-750/` |
| Original QAMPARI checkpoint (April) | `checkpoints/qampari/enc_trained_qampari_infly_multi_finetuned_steps5000_t0.05_lr0.00001_ws200_bs50_gradchkpt_refiltered_5to8_multi_hungarian_with_detach/checkpoint/best_model/` |
| Original AmbigQA checkpoint (May) | `checkpoints/ambigqa/enc_trained_ambigqa_infly_multi_finetuned_steps800_t0.05_lr0.00001_ws30_bs50_hungarian/checkpoint/best_model/` |
| Retrieval results | `results/finetuned/qampari/ddp_step250/qampari.jsonl`, `results/finetuned/ambigqa/ddp_step750/ambigqa.jsonl` (plus `with_detach_control/`, `multi_hungarian/` for the originals) |
| Training logs | `sbatch_outputs/qampari_joint_repro_ddp.out`, `sbatch_outputs/ambigqa_joint_repro_ddp.out`; wandb `run-20260923_154815-rm6cvdbs` (QAMPARI), `run-20260924_111231-g10235a3` (AmbigQA); originals `run-20260429_212024-fsrih0ho`, `run-20260507_145357-c9q43js0` |

Corpus embeddings (`wikipedia_embeddings/{qampari/ddp_step250,ambigqa/ddp_step750}/`, 75 GB each) can be
regenerated from the checkpoints and are not required to keep.

## 7. Related runs (for context)

| Run | Setting | QAMPARI MRecall@100 | AmbigQA MRecall@100 |
|---|---|---|---|
| FSDP repro (fp32 master weights) | `accelerate_config_2gpu_fsdp_nooffload.yaml` | 0.00 (steps 250/500/2000) | 72.67 best (step 150), 70.01 final |
| 1-GPU rerun (pure bf16, global batch 50, uncompressed schedule) | `accelerate_config_1gpu.yaml` | 20.72 (step 250) | — |
| Untrained base model | — | 12.24 | 72.55 |
