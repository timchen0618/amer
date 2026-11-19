# Repo organization proposal (session F, 2026-11-18)

**Proposal:** keep every data directory where it is (checkpoints/, wikipedia_embeddings/, data/, results/, paths are
baked into hundreds of scripts and logs). Reorganize only code, launch scripts and logs, in three steps. The first
step breaks nothing. The other two move files, so they run only when no driver is alive. Nothing here is done yet.

## Today

- The repo root holds 106 entries: the paper's main stack (train.py, train_distributed.py, src/, gen_ret_and_eval.py,
  data_creation/), the shared retrieval scripts (gen_embed_new.py, retrieval_inf.py, retrieval_base.py, eval.py),
  ~15 one-off scripts (plot_*.py, aggregate*.py, bbb.py, remove.py, mess_with_docs.py ...), ~12 sbatch files, and
  stray logs (slurm-*.out, retrieve_base_model.out, single_run_output.log).
- `training/inf_retriever/` is the INF-Retriever stack that all audit work since September ran on. Its `tools/` is
  55 files mixing the run pipeline, evaluation, analysis, data prep, retired launchers and disk tools.
- Three sbatch output folders (sbatch_outputs, sbatch_outputs_eval, sbatch_outputs_qampari) plus sbatch_jobs*,
  sbatch_configs, slurm_scripts.
- `reports/` mixes dated audits, the objective-audit folder (PLAN.md + per-session reports) and weekly summaries.

## Target layout

| Path | What goes there | From |
|---|---|---|
| `amer/` | The paper's main stack, frozen (it reproduces the paper numbers; see paper-pipeline validity notes) | train.py, train_distributed.py, src/, gen_ret_and_eval.py, test.py, data_creation/, scripts/ |
| `retrieval/` | Shared corpus embedding, retrieval and scoring entry points | gen_embed_new.py, retrieval_inf.py, retrieval_base.py, eval.py, eval_utils users |
| `training/inf_retriever/` | Stays (name kept: every driver, checkpoint opt and log refers to it) | — |
| `…/inf_retriever/pipeline/` | Run drivers and job scripts | f_run.sh, f_queue.sh, phaseC_train.sbatch, retrieve_ckpt.sbatch, full_eval_queue.sh, gen_embed_ckpt.sbatch, kagg_*.sbatch |
| `…/inf_retriever/eval/` | Scoring of finished retrievals | kagg_eval.py, fusion_eval.py, fusion_combo*.py, perstep_eval.py, phaseC_collect_scores.py, weekly_collect.py, weekly_render.py |
| `…/inf_retriever/analysis/` | Embedding-space and seed analyses | geometry_eval.py, geometry_compare.py, geometry_seed_summary.py, target_coverage.py, seed_report.py, objective_probe*.py, param_drift.py, copy_stats_offline.py, f0_bounds.* |
| `…/inf_retriever/data_prep/` | Split, dev-set and hard-negative builders | phaseA_*.py/.sh, build_seed_split.py, dev_distance_subsets.py, phaseC_mine_hard_negatives.py, phaseC_cheap_coverage.py |
| `…/inf_retriever/storage/` | Disk tools | slim_all.py, slim_retrieval.py |
| `…/inf_retriever/legacy/` | Retired launchers, kept for the record | abl_launch.sh, abl2_launch.sh, grid_launch.sh, f2_launch.sh, j_*.sh, phase_b_pipeline.sh, phaseC_run.sh, cheap_eval_ckpt.sbatch, eval_drift_ckpt.sbatch, clean_base_retrieval.sbatch |
| `slurm/` | Root-level sbatch files and job folders | gen_embed*.sbatch, retrieve_*.sbatch, sbatch_jobs*, sbatch_configs, slurm_scripts, submit_eval_jobs.sh |
| `analysis/` (exists) | Root plotting and stats scripts | plot_*.py, aggregate*.py, sig_test.py, similarity.py, oracle.py, track.py |
| `archive/` | Dead one-offs, untouched | bbb.py, remove.py, mess_with_docs.py, add_prefix.sh, write_retrieve.py, launch.py, … (each checked unused first) |
| `logs/` (gitignored) | Every job log | sbatch_outputs*, slurm-*.out, *.out / *.log at the root |
| `reports/` | Dated reports + `reports/README.md`: one line per report, newest first, marking the current ones | — |
| `results/README.md` | Which result files are canonical | fusion_S0_*test.json, seed_report.md, geometry_ref/ |

## Steps

1. **Non-breaking (any time):** add `reports/README.md` and `results/README.md`; move dead root one-offs to `archive/`
   after a grep shows nothing imports or calls them; untracked logs into `logs/` (gitignored).
2. **inf_retriever/tools split (no driver alive):** `git mv` into the six subfolders in one commit, rewrite every
   `$T/<tool>` and `tools/<tool>` path (grep for each moved name across the repo, J's and N's worktrees included),
   leave a symlink at the old path for each driver entry point (f_run.sh, f_queue.sh, phaseC_train.sbatch,
   retrieve_ckpt.sbatch) so commands in reports and memory notes keep working.
3. **Root split (no driver alive):** `amer/`, `retrieval/`, `slurm/`, `analysis/` the same way, with symlinks for
   gen_embed_new.py, retrieval_inf.py, retrieval_base.py and eval.py (retrieve_ckpt.sbatch and many old sbatch
   files call them from the root).

Each move step ends with a smoke test before the commit: a 30-step `f_run.sh` training with one dev retrieval,
`fusion_eval.py` and `seed_report.py` on existing results, and `tests/` (slim + sampler tests).

## Open questions

- Should `amer/` stay byte-identical to the paper stack at fe0bd78, or is fixing its known leaks in place wanted?
- J's and N's worktrees (`.claude/worktrees/*`) carry their own copies of these tools: rebase them after the split,
  or leave them as they are until those sessions end?
- The uncommitted account-line edits in 16 sbatch/sh files (not F's): commit, revert, or leave before the moves?
