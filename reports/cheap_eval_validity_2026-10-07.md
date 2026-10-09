# How good is the cheap eval? (2026-10-07)

The cheap eval scores every saved checkpoint on the dev questions with a **reduced corpus**: the base retriever's top-1,000 per dev question, plus all gold passages, plus 100k random passages (`training/inf_retriever/tools/cheap_eval_ckpt.sbatch`; 460k–520k passages, `data/phaseA/corpora/build_info.json`). Retrieval and scoring otherwise match the full eval: the same dev questions, MRecall@100, multi-query with round-robin at k = 5 (QAMPARI) / 2 (AmbigQA), gold-id scoring for QAMPARI and answer-string scoring for AmbigQA. Cheap and full dev data paths match for every pair below.

**Data:** 73 checkpoints with both a cheap and a full dev score: grid 48 (v1 dev), `clean_v3_abl` 14 (v2 dev; QAMPARI 10, AmbigQA 4), pre-grid phase C 11 (leaked training data; shown for completeness). The third-best `clean_v3_abl` QAMPARI checkpoints were still being evaluated and are not included. Per-question MRecall vectors were recomputed from the saved retrieval outputs; they reproduce every reported score exactly. Scripts: `scratchpad/cheap_inv/{collect,analyze,extra}.py`.

## 1. Across runs: a good proxy

Spearman correlation between cheap dev and full dev over checkpoints (n = number of checkpoints):

| set | n | cheap ~ full dev | cheap ~ test | full dev ~ test |
| --- | --- | --- | --- | --- |
| grid QAMPARI | 24 | 0.96 | 0.91 | 0.96 |
| grid AmbigQA | 24 | 0.91 | 0.82 | 0.90 |
| abl QAMPARI | 10 | 0.93 | 0.82 | 0.79 |
| abl QAMPARI multi, against RRF k = 5 dev | 8 | 0.85 | 0.81 (RRF test) | 0.69 |

Cheap dev ranks runs and configurations about as well as full dev does. It is weaker against the RRF k = 5 headline, because the cheap eval scores round-robin.

## 2. Within a run: about a coin flip, but the stakes are small

The question that matters for step selection: among a run's full-evaluated checkpoints, does the cheap-best one have the best full dev?

| set | checkpoint pairs: agree / disagree / tie | cheap-best = full-dev-best | full-dev regret, mean (max) |
| --- | --- | --- | --- |
| grid QAMPARI (2 per run) | 6 / 5 / 1 | 7 of 12 runs | 0.37 (1.40) |
| abl QAMPARI (2 per run) | 1 / 2 / 2 | 2 of 5 runs | 1.00 (2.00) |
| abl QAMPARI multi, against RRF k = 5 dev | 1 / 3 / 0 | 1 of 4 runs | 1.75 (**4.60**, `hps`: cheap prefers step 250, RRF dev prefers 750) |
| grid AmbigQA (steps 150, 400) | 9 / 2 / 1 | 10 of 12 runs | 0.10 (0.60) |

On QAMPARI the cheap eval picks the better of two plateau checkpoints no better than chance. Because the plateau is flat, the cost is usually under one point, but once it is 4.6 points on the RRF headline. Small samples: 5–12 runs per row.

## 3. Why: differences between nearby checkpoints are mostly noise

Paired bootstrap standard error (SE) of the score difference between two checkpoints of the same run, on the same 500 dev questions:

| comparison | median abs. difference | median paired SE | share of differences > 2 SE |
| --- | --- | --- | --- |
| cheap, consecutive steps, grid QAMPARI (n = 108) | 1.10 | 0.99 | 23% |
| cheap, consecutive steps, abl QAMPARI (n = 45) | 1.40 | 1.10 | 27% |
| cheap, consecutive steps, grid AmbigQA (n = 48) | 0.60 | 0.89 | 12% |
| cheap, consecutive steps, abl AmbigQA (n = 20) | 0.40 | 0.63 | 5% |
| **full dev**, the evaluated pairs (n = 34) | 1.00 | 1.09 | 15% |

Most step-to-step differences are within noise, **on full dev as well**. With 500 dev questions no proxy can reliably order neighbouring plateau checkpoints; the cheap eval is not the main limit. Smoothing the cheap curve (mean of each step and its neighbours) does not help: QAMPARI 6 agree / 5 disagree, the same as the raw curve.

## 4. Biases of the reduced corpus

| set | cheap − full dev, points | full-corpus top-100 that is inside the pool | questions solved only in cheap / only in full |
| --- | --- | --- | --- |
| grid QAMPARI single / multi | +10.4 / **+13.1** | 46% / 42% | 10.4% / 0% (single), 13.1% / 0% (multi) |
| abl QAMPARI single | +10.2 | 41% | 10.2% / 0% |
| abl QAMPARI multi / fullsamp | +13.8 / +13.3 | 37% / 38% | |
| abl QAMPARI masked / hps | +10.5 / +10.3 | 37% / 37% | |
| grid AmbigQA single / multi | −4.0 / −3.0 | 74% / 65% | about 1.5–3% / 5.5–5.8% |

- **QAMPARI: cheap overestimates, and more for multi-query.** Every gold passage is in the pool, but most of what the model retrieves from the full corpus (54–63% of its top-100) is not. Those are distractors the cheap eval never shows it. So cheap only ever helps (0% of questions are solved only in full).
  - The overestimate is about 3 points larger for `multi` and `fullsamp` than for `single`, `masked` and `hps`. Cheap dev therefore **understates the single-vs-multi gap by about 3 points** and flatters the variants whose embeddings drift furthest from the base retriever.
  - The size of the overestimate tracks how much of the model's full-corpus top-100 is out of the pool (Spearman −0.38 to −0.48).
- **AmbigQA: cheap underestimates by 3–5 points.** Scoring is by answer string, and the full corpus contains answer-bearing passages that are not in the pool.
  - The cheap-best step is **step 30** (15 steps after warmup, close to the base model) in 9 of 12 grid runs and 4 of 5 abl runs. The pool is built from the base model's top-1,000, so checkpoints close to the base model find its answer-bearing passages.
  - Step 30 was never fully evaluated, so this bias cannot be confirmed. The cheap differences involved (median 0.4–0.6) are within noise.

## 5. Recommendations

1. **Use the cheap eval to screen runs and catch collapse, not to pick steps.** For step selection, fully evaluate a fixed set per run. The proposed top-2 by cheap dev plus the final checkpoint is reasonable: the top-2 sit on the plateau, and the final checkpoint covers late-training changes. Treat full-dev differences under about 2 points as ties, with a fixed tie-break rule.
2. **Score the cheap eval with the headline metric.** Have the cheap retrieval save per-embedding lists and score RRF k = 5 (`kagg_eval.py`). This needs no extra GPU time and removes the round-robin vs RRF mismatch behind the 4.6-point `hps` regret.
3. **Make the QAMPARI pool less base-model-centric.** The full-dev top-100s of the 10 abl QAMPARI checkpoints contain 122k passages outside the current 462k pool (about 24k per checkpoint). Adding them, or the top-100 of a few diverse checkpoints, should shrink both the overestimate and its difference between variants. Cost: about 26% more passages to embed per cheap eval. Build once, on CPU.
4. **Noise is the binding constraint.** With 500 dev questions the paired SE is about 1 point. Doubling the dev set would bring it to about 0.75, but takes training data away; reporting the paired SE with every comparison is free.
5. **AmbigQA:** do not let step 30 win selection without a full eval. Either drop it from the candidates or fully evaluate it once to check the early-checkpoint bias.
