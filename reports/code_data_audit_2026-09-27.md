# Code and data audit, 2026-09-27

Four read-only audits of the INF-Retriever multi-query pipeline: training code, the embed/retrieve/eval pipeline, data splits and tooling, and the raw data. Each finding lists the evidence and whether it was **verified** (measured or reproduced) or **suspected** (from reading the code). Counts marked "re-checked" were verified again independently in the main session.

Already fixed before this audit, and not repeated below:

- stray whitespace on QAMPARI gold passages;
- CSV quoting on random negatives;
- the ignored `normalize` flag.

## 1. Summary

Three problems change what the existing numbers mean:

1. **AmbigQA test questions are in the training data.** 68 of the 827 test questions are in raw train and 65 survive into filtered train. The training script `src/dataset.py` appends AmbigQA dev data to training on purpose, and the test file is that dev set. Every AmbigQA test score so far is inflated by an unknown amount. That covers the single-query bar, the fallback and phase C, and possibly the paper's AmbigQA numbers too.
2. **QAMPARI dev questions are in the training data; QAMPARI test is clean.** The QAMPARI release contains the same example under different ids, and our split is done by row. As a result 1,118 filtered-dev questions and 189 of the 500 dev500 questions also appear in filtered train. Checkpoint selection on dev500 rewards memorization.
3. **Gold lists repeat the same passage.** 10.2% of QAMPARI and 23.5% of AmbigQA training examples list the same gold passage more than once. Multi-query training is then asked to produce the same document twice, which works against its diversity objective.

The evaluation pipeline itself is sound: nothing found biases MRecall or Recall.

## 2. Data

### D1. HIGH: AmbigQA test overlaps train (verified, re-checked)

| AmbigQA file | Questions also in the 827-question test set |
| --- | --- |
| raw train (`data/training/raw/ambigqa/ambigqa_train.jsonl`) | 68 (same ids) |
| filtered train | 65 |
| filtered `ambigqa_2docs` train | 42 |
| filtered dev = phase A dev300 | 2 |

- **Origin:** `src/dataset.py:609-618` builds training data as `train_data + ambig_train_data + ambig_dev_data`, where `ambig_dev_data` is `ambignq-dev_multi_answer_evidence.json`. All 68 hits sit in the last block of raw train (rows 3528–5322, the rows with an extra `answer` field). They share NQ ids with test, but only 6 of 68 have identical answer annotations.
- **Naming:** `data/training/raw/ambigqa/ambigqa_dev.jsonl` is byte-identical to the test file, so its name is misleading.
- **Fix:** drop the 68 test ids and questions from train and from dev300. Existing checkpoints can be re-scored on the 759 clean test questions without any retraining.

### D2. HIGH: QAMPARI duplicate questions leak dev into train (verified, re-checked)

- **Raw train:** 61,911 rows and 61,911 distinct qids, but only 50,173 distinct questions.
  - 11,417 duplicate groups have different qids but identical answers and gold ids: the same example twice. They are almost all `wikidata_comp`, with a median qid gap of 12,600, so probably generated twice upstream.
  - 170 groups are genuinely different examples that share text, e.g. two bands both named "The Radiators".
- **Not ours:** `training/process_qampari_data.py` only drops rows and never copies them, so the duplication comes from the upstream QAMPARI file.
- **The leak:** `data_creation/sample_negatives_and_split.py` shuffles and splits at the row level, so copies land on both sides.

| QAMPARI overlap (question text / id) | Count |
| --- | --- |
| filtered train vs filtered dev | 1,118 / 0 |
| filtered train vs phase A dev500 | 189 / 0 (re-checked: 189 of 500) |
| any train or dev file vs test (`eval_data/qampari.jsonl`, 531 questions) | 0 / 0 |

- **Fix:** deduplicate by (normalized question, gold set) before splitting, then split by question and rebuild dev500. The docstring in `phaseA_make_dev_sets.py:10` claims no overlap and is wrong.

### D3. HIGH: repeated gold passages within an example (verified, re-checked)

| File | Examples with a repeated gold | Repeated gold slots | All golds identical |
| --- | --- | --- | --- |
| QAMPARI train | 10.2% | 3.1% | 160 |
| AmbigQA train | 23.5% | 11.1% | 539 |
| AmbigQA filtered dev | 23.7% | 9.5% | – |
| QAMPARI test | 46.7% | 22.3% | 0 |
| AmbigQA test | 77.0% | 30.7% | 36 |

- **Cause:** golds are grouped per answer, and one passage can support several answers. The training data keeps one passage per answer, so the passage repeats.
- **Evaluation is fine:** MRecall counts answer clusters, and one passage may legitimately cover several.
- **Training is affected:** `Dataset` (`src/finetuning_data.py:98`) uses the list as is; only `SampleDataset` deduplicates.
  - Multi-query training is asked to produce the same document several times.
  - Under the `hungarian` loss, the repeated copy also sits in the softmax denominator with an identical score. That caps the probability of the right target at 1/2 or less.
- **Fix:** deduplicate golds by id (or title + text) in the loader and recompute `gold_counts`.

### D4. MEDIUM: hard-negative false negatives (verified)

- **Clean:** no hard negative is a gold passage by id, and none has the answer string in its title.
- **Same-article chunks:** 3.6% of QAMPARI hard negatives (31,602 of 870,600) are another chunk of a gold passage's article. For AmbigQA it is 10% (14,185 of 142,296).
- **Answer-entity mentions:** 2.9% of QAMPARI hard negatives mention the answer's source entity.
- **Why AmbigQA is worse:** 54% of AmbigQA train examples have no gold ids, so only the exact-token answer filter protects them.
- **Empty lists:** only 3 of the 29,023 QAMPARI examples ended up with no hard negatives.
- **Code robustness:** `zip(fin, mined)` would silently truncate if the mining file were short. A per-line question assert does guard the ordering.
- **Random negatives are not clean either:** 1.8% of QAMPARI random negatives (12,726 of 725,575) and 0.8% of AmbigQA ones (981 of 118,600) contain an answer string. The random-negative sampler filters same-title passages but not answers.
- **Fix:** in `phaseC_mine_hard_negatives.py`, reject candidates whose title matches a gold title or an answer entity, as the random-negative sampler already does for titles.
- **Correction (2026-09-28): the answer filter looking at text only is not a weak spot.** It was flagged because `has_answer` checks only `text`, not `title`. But in the chunks_v5 corpus every passage's text already starts with its title: 740,704 of 740,704 QAMPARI and 116,131 of 116,131 AmbigQA hard negatives, and every random negative. The text check therefore already covers the title.
  - Measured on `data/training/clean_hn`: 0 hard negatives have an answer in the title but not in the text.
  - Checking `title + " " + text` instead would catch only 21 more (0.003%). Twenty are coincidences at the boundary where the title repeats, e.g. "Osborne Brothers Osborne Brothers…" matching the answer "Brothers Osborne". One is a real miss, "Zhang Ziyi" for the answer "Ziyi Zhang"; catching it would need name-order handling, not a title check.
  - AmbigQA scoring (`src/eval_utils.py`) also checks text only, so mining and evaluation agree.
  - This holds only while passage text begins with the title. A corpus without that convention would reopen the gap.

### D5. MEDIUM: dev500 does not look like the QAMPARI test set (verified)

- **Gold counts match:** 5/6/7/8 golds = 201/127/93/79 in dev500 vs 226/135/90/80 in test.
- **Question types don't:**

| Question type | dev500 | test |
| --- | --- | --- |
| `wikidata_intersection` | 5% | 25% |
| `wikidata_comp` | 44% | 22% |
| `wikitables_simple` | 0 | 15 questions |

- **Different source splits:** dev500 comes from QAMPARI's train split; test is the 5–8-gold subset of QAMPARI's dev split.
- **Different gold structure:** dev500 keeps one gold per answer, while test answers often have two.
- **Effect:** dev and test scores are not comparable in absolute terms, and the rankings may differ.

### D6. LOW

- 10 raw QAMPARI rows have an empty question (9 in filtered train, 1 in filtered dev).
- 683 of the 827 AmbigQA test golds have no id and no title. This is harmless, since AmbigQA is scored by answer string.
- Raw AmbigQA train has 2 duplicate questions ("Who wrote wake me up when it's all over?", "Who died in the plane crash grey's anatomy?"). Each pair has different ids and different answer annotations, probably from merged sources.
- The QAMPARI "test" set is the 5–8-gold subset of QAMPARI's dev release, not QAMPARI's official test split.
- `data/amer_data/*_train.jsonl` and the files in `data/final_eval_data/` are byte-identical copies of the raw and eval files. `data/amer_data/eval_data/ambigqa_2docs.jsonl` (474 questions) is a subset of the AmbigQA test set, and 37 of its questions are in filtered train.

## 3. Training code (`training/inf_retriever/`)

### T1. MEDIUM: the multi-query loss treats an example's other golds as negatives (verified)

- `phaseC_train.sbatch` passes `--loss_fn hungarian`.
- `src/inbatch.py:96-97` takes the log-softmax over the whole candidate pool before matching. So the other k-1 golds of the same example sit in the denominator, pushing outputs away from them.
- The option's own help text calls this the "same-example false-negative issue". A masked variant, `hungarian_masked`, already exists.
- Single-query runs use plain contrastive loss, so the two modes also differ in loss family.

### T2. MEDIUM: teacher forcing conflicts with Hungarian matching (suspected)

- Without `--full_sampling`, the sampling rate is step / total_steps. On average, about half of the fed-back embeddings are gold embeddings in shuffled order (`src/inbatch.py:580-584`).
- The loss matches each output to whichever gold fits best, not to the one fed at that position. So a later step can be scored against a gold already in its own context, and the model can learn to copy it.
- That situation never arises at inference, where `generate()` always feeds back its own outputs.
- **Options:** `--full_sampling`, or feed back the matched gold.

### T3. MEDIUM: query instruction differs between training and inference (verified; measured effect negligible on one checkpoint)

- Training (`src/finetuning_data.py:79`): "Given a query, retrieve relevant passages that answer the query".
- Retrieval (`retrieval_inf.py:117, 231`) hard-codes "Given a *web search* query, …". The flag `--query_instruct_task` defaults to the training string but is never read.
- **Measured** on `multi_fix_lr1e-5/step-2500` with 150 test queries: MRecall@100 82.67 vs 82.00, about 1 query in 150.
- **Fix:** read the flag in both embedding functions. `--question_maxlength` in `retrieval_inf.py` is also never read.

### T4. LOW, but a trap: FSDP is one unit for the whole model, which hides a rank-mismatch deadlock (verified)

- accelerate's per-layer wrapping looks for `_no_split_modules` on the outer model, which the wrapper class does not define. So FSDP wraps the whole model as a single unit.
- The two GPUs get batches with different gold counts k in about 70% of steps, so they run different numbers of generation steps. With the model as one unit this works.
- A 2-GPU test with per-layer wrapping (k = 2 vs 5) hung in the backward pass. If wrapping is ever "fixed", multi-query training will deadlock.
- **Fix:** make both ranks draw the same k per step, or document that the single-unit wrapping is deliberate.

### T5–T9. LOW

- **T5.** `--max_positive_documents` does nothing (its code is commented out at `src/finetuning_data.py:85-88`). Multi-query trains on all golds: k = 5–8 for QAMPARI, 2–5 for AmbigQA.
- **T6.** Loss logits are computed in bf16. At temperature 0.05, logits of 10–16 have bf16 steps of 0.06–0.125. Casting to fp32 before the einsum would fix it.
- **T7.** `Dataset` would shard by rank if the process group already existed. `prepare_data` happens to run before `build_accelerator`, so every rank currently sees the full data, which the logs confirm. Reordering those calls would silently cut the data to a quarter per rank.
- **T8.** `CosineScheduler` uses `math` without importing it, and with `accumulation_steps > 1` the scheduler would step every micro-step. Neither is used now.
- **T9.** 1.7% of QAMPARI examples share a gold with another example in the same batch, which then counts as an in-batch negative.
- **T10.** The `hungarian` loss logs no training accuracy, so multi-query training curves show loss only.

**Checked and correct:**
- the cross-GPU gradient through `VarsizeGather`;
- label offsets in all losses;
- tokenizer, EOS, padding, position ids and last-token pooling, which match between training and inference;
- `generate()`, which builds inputs the same way as training;
- teacher embeddings, which are detached as intended;
- `drop_last` with batches grouped by gold count;
- Qwen2 has no dropout modules, so `--dropout` is a no-op.

## 4. Embedding, retrieval and evaluation

Nothing biases MRecall or Recall.

- **E1. LOW:** nDCG and mAP for multi-query runs re-sort documents by score instead of using the round-robin order (`src/eval_utils.py:175`). We don't report them.
- **E2. LOW, latent:** `--save_or_load_index` with more than one FAISS shard saves shard 0's index and loads it for every shard, so 15/16 of the corpus is never searched (`src/retrieval_utils.py:145-157`). No wrapper uses this flag.
- **E3. LOW:** 25 of the 25.86M corpus titles ("NaN", "N/A", "None") are embedded as the string "nan", because of pandas NA handling in `gen_embed_new.py:208`.
- **E4. LOW:** retrieval output files keep 16 times more candidates than needed, since results are never truncated after merging shards. Scores are unaffected.
- **E5. LOW, latent:** FAISS `-1` padding ids are not filtered. This can't happen at current sizes.

**Checked and correct:**
- shard boundaries, with no passage dropped or duplicated;
- corpus parsing (25,856,230 records, no malformed rows, no duplicate ids);
- fp16 inference vs fp32 training (cosine ≥ 0.990);
- exact `IndexFlatIP` search;
- round-robin deduplication, which always yields 100 unique documents;
- the MRecall and Recall definitions;
- AmbigQA answer matching.

## 5. Experiment tooling (`training/inf_retriever/tools/`)

- **X1. MEDIUM:** `phaseC_run.sh` prunes checkpoints by cheap-dev rank. Its protection check matches queued checkpoints by exact relative path (`grep " $ck "`). A queue line with an absolute path, a `./` prefix or a trailing slash would not match, so that checkpoint could be deleted. `full_eval_queue.sh` then skips it silently. Current queue files are unaffected.
- **X2. MEDIUM, suspected:** if a run name is reused, `phaseC_run.sh` can mistake the old training log's "Saving model" lines and old `eval_metrics.txt` files for new ones.
- **X3. LOW:** `full_eval_queue.sh` marks a checkpoint done once its test metrics exist, so a failed dev eval is never retried. Failures of the eval step are not detected.
- **X4. LOW:** `phaseC_collect_scores.py` labels runs from name substrings, and would crash on a run directory without `_lr` in its name.

**Checked and correct:**
- the reduced corpora contain every gold (3,585 of 3,585 for QAMPARI test, 2,926 of 2,926 for dev500);
- ids have no stray whitespace;
- base-retriever results line up with their query files;
- the top-1,000 is taken after the cross-shard sort;
- `EXTRA` flags override the sbatch defaults.

## 6. Proposed actions

**Fix before the clean grid** (they change the data or the training objective):

| # | Action | Addresses |
| --- | --- | --- |
| 1 | Remove the 68 AmbigQA test questions from train and dev300 | D1 |
| 2 | Deduplicate QAMPARI by (question, gold set), split by question, rebuild dev500 and its reduced corpus | D2, D6 |
| 3 | Deduplicate golds per example in the loader | D3 |
| 4 | Re-mine hard negatives with the same-title and answer-entity filters | D4 |
| 5 | Make retrieval use the training query instruction | T3 |
| 6 | Harden the prune path match and reset state for reused run names | X1, X2 |

**Decisions** (they change the method):

| # | Question | Addresses |
| --- | --- | --- |
| 7 | Loss for multi-query: `hungarian` (current) or `hungarian_masked` | T1 |
| 8 | Teacher forcing: keep scheduled sampling, use `--full_sampling`, or feed back the matched gold | T2 |
| 9 | Dev set design: match test's question-type mix | D5 |

**Can wait:** T4 (document it), T5–T9, E1–E5, X3–X4.

**Beyond this pipeline** (the paper's main multi-query path; not measured here):

- `src/dataset.py:609-618` appends AmbigQA dev, which is the test set, to training.
- `data_creation/generate_embeddings.py` and `data_creation/gen_embed_for_auto_training.py` embed QAMPARI golds from the raw train file, stray whitespace included, while negatives come from the clean corpus. `gen_embed_for_auto_training.py` takes its dev split from the clean raw dev file, so its train and dev are inconsistent.
- `data_creation/create_doc_enc_dataset.py` has the same whitespace pattern. Its `is_same_doc` (line 125) compares titles exactly, so a gold's title `"X "` never matches its corpus article `"X"`. Negatives from the gold's own article are therefore not rejected; `generate_embeddings.py:168` and `gen_embed_for_auto_training.py:271` share the flaw.
- `training/contriever` and `training/qwen3` join title and text without stripping, so they are affected if fed QAMPARI train data.

The paper's QAMPARI and AmbigQA numbers may be affected.

## 7. Decisions (2026-09-27)

| Item | Decision |
| --- | --- |
| Fixes 1–6 | Do all of them before the 24-run grid. Deduplicate repeated golds in the training data only, report how the gold count per example changes, and drop AmbigQA training examples left with a single gold. |
| Dev sets (D5) | Keep random sampling (option c): matching test's question-type mix would peek at the test distribution. Both datasets get a final dev set of exactly 500 questions after all filtering and deduplication; QAMPARI's 3,000-question dev split is dropped. |
| Loss (T1) | Keep `hungarian` for the 24-run grid; ablate objectives later. |
| Teacher forcing (T2) | Keep scheduled sampling for the grid; ablate the sampling rate later. |
| FSDP wrapping (T4) | Document only. Do not enable per-layer wrapping for multi-query training without first making both ranks draw the same gold count per step. |
| Re-scoring on clean AmbigQA test questions | Not done: everything will be re-run. |
| Paper pipeline | Out of scope for now. |

## 8. Fix candidates for the next round (after the 24-run grid)

These change the training data, so they wait until the grid finishes. The grid (launched 2026-09-28) trains on the clean splits as built.

| # | Candidate | Size | What it takes |
| --- | --- | --- | --- |
| N1 | **Filter answer-containing random negatives.** The random-negative sampler rejects same-title passages but never checks for answers. | 1.26% of QAMPARI and 0.88% of AmbigQA random negatives in `data/training/clean` contain an answer string (earlier, on the pre-audit data: 1.8% / 0.8%). | Apply the same `has_answer` check the hard-negative miner uses, and resample the rejected negatives. |
| N2 | **Recover QAMPARI examples dropped by the raw 5–8 gold filter.** The filter counted raw gold slots. Rows with more than 8 slots but only 5–8 unique passages were excluded, although they qualify after deduplication. | 556 new training examples (+2.2%) under "5–8 unique golds"; 753 under 2–8 and 894 under 1–8. None overlaps test or the clean dev set. They skew to `wikidata_simple` (63% vs 46% overall), with about 1.6 answers per passage. | Sample random negatives for them, rebuild the QAMPARI train split (dev unchanged), and re-mine hard negatives. |
| N3 | **QAMPARI training examples with fewer than 5 unique golds.** 1,421 examples, 143 of them with a single gold, remain in train after deduplication. They were kept by decision; the equivalent AmbigQA examples with one gold were dropped. | 5.7% of QAMPARI train. | Decide whether to apply a minimum (e.g. `--min_train_golds 2`), consistently with N2's rule. |
