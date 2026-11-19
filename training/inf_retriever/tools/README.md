# training/inf_retriever/tools/

Each script listed once, by job. Run everything from the repo root, and heavy work through srun / sbatch.

## Known traps
- **Old dev default.** `fusion_eval.py` and `perstep_eval.py` default to the old 500-question dev (`data/phaseA/qampari_cleanv2dev500.jsonl`). For S0 runs pass `--dev data/phaseA/qampari_clean_v3_dev_s0.jsonl` to `fusion_eval.py`. `perstep_eval.py` has no such option, so it fails on S0 runs: harmless, ignore it.
- **`full_eval_queue.sh` workers** must be started with `PER_STEP=1 DEV_TAG=cleanv2`, copied from the running workers. The defaults use the leaky old dev.
- **The paper's code is not here.** The root `train.py` / `src/` / `gen_ret_and_eval.py` are March rewrites that cannot rerun the paper. Use tag `paper-repro-v1` (see `reports/paper_repro_2026-10-23/REPORT.md`).
- **Don't edit a driver while it runs.** Bash reads a running script from disk as it goes.

## Run pipeline (current)
| Script | Role |
|---|---|
| `f_queue.sh` | Session F's queue: one line per run (name, flags, train script, env). Launches each run once |
| `f_run.sh` | One run end to end: train (`phaseC_train.sbatch`), dev retrieval at every saved step, test on the best steps, keep / prune checkpoints, slim outputs |
| `phaseC_train.sbatch` | Training job (2× H200, accelerate FSDP). `CODE_ROOT` trains a worktree's code |
| `retrieve_ckpt.sbatch` | Embed queries for one checkpoint and search the frozen corpus index |
| `gen_embed_ckpt.sbatch` | Embed the corpus with a checkpoint's document encoder, one shard per array task |
| `full_eval_queue.sh` | Older full-eval worker queue (see the trap above) |
| `kagg_retrieve.sbatch`, `kagg_eval.sbatch` | Retrieval and scoring for the k-aggregation study |
| `n_launch.sh`, `n_queue.sh`, `q_queue.sh` | Session N's launcher and queues (Nemotron, Qwen3) |

## Scoring
| Script | Role |
|---|---|
| `fusion_eval.py` | Fuse a multi run's k lists every way (RRF k=60 = headline) and score dev, test and the paper's large / small thirds; `--single` for single-query runs |
| `fusion_combo.py`, `fusion_combo_summary.py`, `fusion_summary.py` | Fusing several checkpoints; summary tables |
| `kagg_eval.py` | Score a multi-query checkpoint for every number of query embeddings k and every aggregation |
| `perstep_eval.py` | Each query embedding's own top-100, scored alone (old dev only) |
| `seed_report.py` | Seed table: `results/phaseC/seed_report.md` |
| `phaseC_collect_scores.py`, `weekly_collect.py`, `weekly_render.py` | Score collection and the weekly report |

## Analysis
| Script | Role |
|---|---|
| `geometry_eval.py`, `geometry_compare.py`, `geometry_seed_summary.py`, `target_coverage.py` | Query-embedding geometry and per-target coverage |
| `objective_probe.py`, `objective_probe_table.py` | Objective-audit D2 / D3 probes |
| `param_drift.py`, `copy_stats_offline.py`, `f0_bounds.py`, `f0_bounds.sbatch`, `phaseC_cheap_coverage.py` | Weight drift, copy-shortcut stats, F0 bounds, cheap coverage |

## Data preparation
| Script | Role |
|---|---|
| `build_seed_split.py` | Seeded 90/10 train / dev split of the clean QAMPARI pool (S0 = seed 0) |
| `dev_distance_subsets.py` | Dev large / small thirds with the paper's thresholds |
| `phaseA_*.py`, `phaseA_*.sh`, `phaseA_validation_manifest.tsv` | Phase A: reduced corpus, dev sets, setup and validation |
| `phaseC_mine_hard_negatives.py` | Hard-negative mining (inf-retriever) |
| `nemo_*`, `qwen3_*` | Session N: Nemotron / Qwen3 corpus and query embedding, hard-negative mining, checks |

## Storage
| Script | Role |
|---|---|
| `slim_all.py`, `slim_retrieval.py` | Shrink finished retrieval outputs (keep ids and scores); see `results/phaseC/full/README_slim.md` |

## Retired launchers (kept for the record, not maintained)
`abl_launch.sh`, `abl2_launch.sh`, `grid_launch.sh`, `f2_launch.sh`, `phase_b_pipeline.sh`, `phaseC_run.sh`, `cheap_eval_ckpt.sbatch`, `eval_drift_ckpt.sbatch`, `clean_base_retrieval.sbatch`, `kagg_study.sh`; session J's `j_launch.sh`, `j_drivers.sh`, `j_deferred.sh`, `j_select.sh`.
