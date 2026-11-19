# Handoff: same F setting, different retriever (2026-11-12)

Goal: rerun session F's current setting with a different retriever of similar size, e.g. `nvidia/Nemotron-3.5-Embed-1B-BF16`, in place of `infly/inf-retriever-v1-1.5b`. Everything else stays the same so the result compares to F's numbers. Written by session F at the user's request. Read this file, then `reports/objective_audit_2026-10-13/F_paper_gap_plan_2026-10-22.md` (results so far) and the Status log at the end of `reports/objective_audit_2026-10-13/PLAN.md`.

## 1. The setting to copy

Repo `/scratch/hc3337/projects/autoregressive`, branch `fsdp-clean-recipe`. Frozen document encoder, QAMPARI, S0 data, 10k steps, temperature 0.05.

| Role | Run (F name) | Queue entry, `training/inf_retriever/tools/f_queue.sh` | Dev / test |
|---|---|---|---|
| Multi base | `S0_10k_onelabel_causal` | line ~99: `$S0 --loss_fn hungarian --one_label_step --force_causal`, script `finetuning_multi_paper.py`, env `$S0ENV $LONG20` | 49.78 / 32.77 (RRF k=5, step 9000) |
| Single base | `S0_single_10k_2step_causal` | line ~84: `$S0 --one_label_step --force_causal`, same script and env | 51.16 / 35.22 (MRecall@100, step 10000) |

- The variables `S0`, `S0ENV`, `BASE`, `LONG` and `LONG20` are defined at the top of `f_queue.sh`. `BASE` (always prepended) adds `--negative_hard_ratio 0.5 --freeze_doc_encoder`.
- How a run goes: `f_queue.sh` launches `tools/f_run.sh <mode> 1e-5 <run> "<BASE extra>"`. The driver submits `tools/phaseC_train.sbatch` (2× H200, FSDP), runs full-dev retrieval on every saved step, keeps the top 2 by dev, then tests the top 2, the final step and the lowest-dev-loss step. `f_run.sh`'s header comment documents all of this.
- Selection rule (user): test at the dev-selected step. A difference under 2 points is a tie, and ties go to the simpler setting. A change doesn't count if dev drops by 2 or more. Report the large/small distance thirds with `tools/fusion_eval.py` (results/phaseC/fusion_S0_test.json, fusion_S0_single_test.json).

## 2. Where inf-retriever is hard-wired (the real work)

Swapping the model is not a flag change. Check each item, and confirm the new model's facts from its model card / config before coding:

1. **Corpus embeddings.** The frozen doc encoder retrieves against a precomputed corpus embedding, `wikipedia_embeddings/qampari/shared_frozen_base_512` (`EMB=` in `f_run.sh`; 32 pickle shards, 75 GB, made with inf-retriever at chunk length 512). A new retriever needs the whole corpus re-embedded with it (`gen_embed_new.py` / `gen_embed.sbatch`, see CLAUDE.md Stage 2), into a new directory, and `f_run.sh` pointed at it (e.g. an env override, not an in-place edit; see section 4).
2. **Model id.** `--model_path infly/inf-retriever-v1-1.5b` is hard-coded in `tools/phaseC_train.sbatch` (line ~61); `src/inference_utils.py` (`load_retriever`, ~line 28 and ~203) falls back to the infly id and tokenizer.
3. **Embedding size 1536.** `--projection_size 1536` (options.py default, `tools/retrieve_ckpt.sbatch` ~line 49) and `hidden_size = 1536` in `training/inf_retriever/src/inf_retriever.py`. The multi model feeds embeddings back as input tokens, so the LM hidden size and the embedding size must match or go through a projection (`--io_proj`).
4. **Pooling, prompt, loading.** inf-retriever uses last-token pooling and an instruct-format query prompt (`--query_instruct_task`, `embed_queries_single`). Check the new model's pooling, query/document prompts, normalization, and whether it needs `trust_remote_code` or a newer `transformers` than the `nli` overlay has.
5. **Hard negatives.** `clean_v3_hn` / `clean_v3_hn_s0` negatives were mined with inf-retriever. Reusing them is reasonable, but state it as a choice in the report.

## 3. Resources and rules (from the user)

- Slurm: always `--account=torch_pr_152_echoi`; never `--partition`; H200 via `--gres=gpu:h200:N`; retrieval jobs `--constraint="h200|l40s|b200" --gres=gpu:1 --comment="preemption=yes;requeue=true"`. Never run Python on the login node (srun/sbatch).
- GPUs: 16 per user in total, shared with session F's queue (F uses up to 8). Ask the user how many this work may use before starting.
- **Disk: /scratch is at 4.43 / 5.00 TB (`myquota`).** A second corpus embedding is ~75 GB plus checkpoints; check with the user before large writes. No deletions without approval, and never delete checkpoints that have `KEEP_fallback_recipe.txt`.
- Test sets stay unchanged. Don't edit J's files: `inbatch.py`, `finetuning_multi.py`, `options.py`. F's own entry script is `finetuning_multi_paper.py`.
- Coordinate with sessions F, J and R only through one-line entries in PLAN.md's Status log ("- YYYY-MM-DD <id>: ...").

## 4. Gotchas learned the hard way

- **Never edit `f_run.sh` or `f_queue.sh` in place while drivers run from them**: bash reads scripts by byte offset, so an edit can break running drivers. Write a new file and `mv` it over (new inode).
- New runs go in `f_queue.sh`'s QUEUE (it skips runs that have a `results/phaseC/<run>.trainjob`), started with `MAXRUN=3 setsid nohup bash training/inf_retriever/tools/f_queue.sh >> results/phaseC/f_queue.out 2>&1 < /dev/null & disown`. Never start a second instance. Stop it with `kill <pid>`, not `pkill -f` (that killed the agent's own shell).
- Restarting the same `f_run.sh` command resumes a run from its `.trainjob` file. A job that failed before step 1000 needs that file moved aside first.
- Login nodes don't share a process table, and a login node can reboot and kill every driver (Slurm jobs survive). Before relaunching, `hostname`, then ssh-check the other login nodes for `tools/f_run.sh` processes.
- Slurm can be slow: an sbatch reply timeout can still create the job, and squeue can time out. Both scripts now guard against this.
- S0 runs need `S0ENV` (`F_DEV`, `F_DATA_ROOT`); without it they use the old dev set. The driver's "per-embedding scoring failed" error on S0 runs is harmless (RRF scores are unaffected).
- Do a short smoke run first (`STEPS_OVERRIDE`, `WARMUP_OVERRIDE`, `FREQ_OVERRIDE`, `SAVE_AT_OVERRIDE` env vars on `phaseC_train.sbatch`) and check memory: the 1.5B model peaks at ~46 GB per GPU.

## 5. Suggested order

1. Confirm the new model's dimension, pooling, prompts and loading (item 2.3–2.4); smoke-load it on a GPU node.
2. Ask the user about GPU and disk budget; re-embed the corpus with the new model.
3. Make the retriever a parameter (model id, embedding dir, projection size) in F-owned or new files, defaulting to inf-retriever so F's runs are unchanged.
4. Smoke-train single and multi; then the two baselines in section 1; report against F's numbers with thirds.
