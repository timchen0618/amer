# QAMPARI multi-query: week of Oct 5–11

QAMPARI only (AmbigQA is on hold). Week of 2026-10-05 to 2026-10-11. Numbers as of 2026-10-11 19:34; runs still in progress are marked pending: hps 0.1, hps + full sampling, hps + saturate 50%, C single, C multi, A single, A multi, B single, B multi.

## Summary

- **Single-query is still the best QAMPARI retriever on test.** `single` scores 53.30 test (64.80 dev). The best multi-query run is round 1's `masked`: 47.65 test, 65.8 dev (k = 5, RRF). It is higher on dev but 5.65 points lower on test. None of the 12 finished runs closes that gap.
- **Every objective that helped made the model behave more like a single-query retriever.** `masked`'s five queries are almost the same vector (cosine 0.94). `hps 0.5` is best with its first query alone: 54.05 test at k = 1, the only multi-query number above `single`. Adding its other four queries lowers the score.
- **Scored one at a time, the embeddings of the diverse runs are weak but complementary.** Each finds 27–43% of a question's golds (`single`'s one query: 77.6%); fused, they reach 38–43 test. `masked`'s five embeddings are each as good as the fused result, and `hps` has one strong embedding plus weak ones. No run has embeddings that are both.
- **Forcing the queries apart does not help recall.** The diversity penalty lowers the query–query cosine from 0.94 to 0.63, as intended, but test falls from 47.65 to 44.63 (λ = 2).
- **Hungarian multi-query training pulls a question's golds apart in document space.** Gold–gold cosine is 0.50–0.53 against 0.59 for `single`, and the late checkpoints that collapse drop to 0.42–0.47. The masked loss keeps the golds together (0.57–0.61) and gives the best multi-query run, but that is not enough on its own: the diversity-penalty runs keep them together too and score 3–4.5 points lower.
- **Earlier full sampling, earlier saturation and a higher learning rate all hurt.** The sampling runs peak mid-training, then their first query collapses (test at k = 1 falls below 1 by step 2500). LR 3e-5 is worse than 1e-5 for both modes.
- **Measurement fixes.** The earlier multi-query 44.26 was inflated by 44 QAMPARI test questions that leaked into the v1 training data. Cheap dev cannot pick the best checkpoint within a run, so every run is now selected by full dev over the top 2 cheap-dev checkpoints plus the final one.

## What was done this week

| date | work |
| --- | --- |
| Oct 5–7 | Round 1 (`clean_v3_abl`) on QAMPARI: `single`, `multi`, `masked`, `full sampling`, `hps 0.5`, each fully evaluated at three checkpoints, with k = 1–5 and round-robin/RRF from one retrieval. |
| Oct 6 | Headline metric fixed as MRecall@100 at k = 5 with RRF; round-robin and other k are secondary. |
| Oct 7 | Found the v1 test leak (44 of 531 QAMPARI test questions), which explains the drop from the grid's 44.26. |
| Oct 7 | Cheap-eval validity study: good for ranking runs, close to a coin flip for picking steps within a run. Selection protocol changed. |
| Oct 7 | Round 2 code: sampling saturation, masked + diversity penalty, RRF-based cheap selection, keep the final checkpoint. |
| Oct 7–11 | Round 2, wave 1: eight runs (hps weights, diversity penalty, sampling saturation, LR 3e-5). All evaluated except `hps 0.1`. |
| Oct 9–10 | Wave 3 code: frozen document encoder under FSDP (fixed by making the frozen copy an FSDP-ignored module, no DDP fallback), causal query attention for every model class, one shared corpus embedding for all frozen-encoder checkpoints. |
| Oct 10–11 | Wave 3 launched (cells C, A, B). Wave 2 launches automatically once the hps weight is decided. |
| all week | Operations: /scratch quota incident (Oct 8), B200 nodes back with automatic requeue, disk-aware launch gates, an eval-queue bug that flagged good results as failed. |

## Main results

Each run at the checkpoint with the best **full dev** score; test is never used for selection. Multi-query runs are scored at k = 5 with RRF; single-query runs use their single query. MRecall@100 on the 500 dev and 531 test questions.

| group | run | step | dev | test | test, k = 1 | q–q cos | g–g cos | q–q − g–g | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | single | 2000 * | 64.80 | 53.30 | – | – | 0.59 | – | done |
|  | multi | 1500 * | 61.2 | 38.42 | 16.57 | 0.67 | 0.50 | +0.17 | done |
| objective | masked | 750 * | 65.8 | 47.65 | 49.34 | 0.94 | 0.57 | +0.38 | done |
|  | masked + div 0.5 | 750 | 61.4 | 43.13 | 48.59 | 0.84 | 0.58 | +0.25 | done |
|  | masked + div 2 | 750 | 60.0 | 44.63 | 45.01 | 0.63 | 0.61 | +0.02 | done |
|  | hps 0.5 | 750 * | 63.0 | 46.14 | 54.05 | 0.64 | 0.53 | +0.11 | done |
|  | hps 0.25 | 1000 | 62.8 | 40.49 | 46.52 | 0.65 | 0.53 | +0.12 | done |
|  | hps 0.1 |  |  |  |  |  |  |  | full evals |
| sampling | full sampling | 750 * | 56.8 | 38.04 | 9.60 | 0.60 | 0.52 | +0.09 | done |
|  | saturate 25% | 500 | 57.2 | 41.24 | 7.53 | 0.64 | 0.53 | +0.11 | done |
|  | saturate 50% | 1000 * | 60.4 | 43.31 | 6.59 | 0.66 | 0.52 | +0.15 | done |
| learning rate | single, LR 3e-5 | 1250 * | 61.40 | 48.21 | – | – | 0.61 | – | done |
|  | multi, LR 3e-5 | 1750 * | 55.8 | 39.92 | 10.36 | 0.67 | 0.52 | +0.15 | done |
| combination (wave 2) | hps + full sampling |  |  |  |  |  |  |  | starts after hps 0.1 |
|  | hps + saturate 50% |  |  |  |  |  |  |  | starts after hps 0.1 |
| doc encoder × attention (wave 3) | C single |  |  |  |  |  |  |  | full evals |
|  | C multi |  |  |  |  |  |  |  | full evals |
|  | A single |  |  |  |  |  |  |  | cheap evals |
|  | A multi |  |  |  |  |  |  |  | waiting for GPUs |
|  | B single |  |  |  |  |  |  |  | waiting for GPUs |
|  | B multi |  |  |  |  |  |  |  | waiting for GPUs |

- **q–q cos**: mean cosine between a question's 5 query embeddings (test). Blank for single-query runs.
- **g–g cos**: mean cosine between the gold passages of the same question, in that model's own document space (test). Training changes the document encoder, so this differs between runs.
- **q–q − g–g**: positive means the queries are more alike than the golds they should cover (collapse); near 0 means they are spread like the golds.
- \* the selected step beats another evaluated step by less than 2 dev points (about two paired standard errors): treat as a tie.

### Test score against the number of queries

*Figure (see the HTML page): Test MRecall@100 at k = 1–5 (RRF over the first k queries), selected checkpoints. Runs whose queries differ (`multi`, the sampling runs) need all five queries; runs that act like a single query (`masked`, `hps`) are flat or fall with k. None reaches `single` at k = 5.*

### Each query embedding on its own

The same retrieval, scored with one embedding's top 100 alone (embedding 1 = the first generated). Round 1 deleted the per-embedding lists of most selected checkpoints, so `multi`, `masked` and `hps 0.5` use their other evaluated checkpoint (marked); it is within 2 dev points of the selected one.

*Figure (see the HTML page): Test MRecall@100 of each query embedding used alone. MRecall needs every gold of a question in the top 100, so an embedding that targets one gold scores low by design; Recall@100 (second table) shows how much each one covers.*

| run | step | emb 1 | emb 2 | emb 3 | emb 4 | emb 5 | all 5, RRF |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multi | 1750 (selected 1500) | 10.73 | 4.33 | 3.01 | 1.51 | 7.72 | 39.36 |
| masked | 1000 (selected 750) | 47.46 | 49.15 | 48.59 | 48.02 | 46.14 | 48.59 |
| masked + div 0.5 | 750 | 48.59 | 42.18 | 39.92 | 35.97 | 37.10 | 43.13 |
| masked + div 2 | 750 | 45.01 | 42.94 | 39.74 | 34.84 | 33.90 | 44.63 |
| hps 0.5 | 1000 (selected 750) | 51.98 | 23.35 | 13.37 | 11.30 | 11.49 | 47.27 |
| hps 0.25 | 1000 | 46.52 | 10.73 | 10.55 | 8.10 | 14.31 | 40.49 |
| full sampling | 750 | 9.60 | 3.01 | 1.88 | 1.88 | 1.32 | 38.04 |
| saturate 25% | 500 | 7.53 | 4.71 | 3.20 | 3.39 | 6.21 | 41.24 |
| saturate 50% | 1000 | 6.59 | 7.53 | 4.90 | 5.08 | 4.52 | 43.31 |
| multi, LR 3e-5 | 1750 | 10.36 | 11.68 | 4.90 | 2.45 | 3.20 | 39.92 |
| single (reference) | 2000 | 53.30 |  |  |  |  |  |

Recall@100 (share of a question's golds found) of each embedding alone, test:

| run | step | emb 1 | emb 2 | emb 3 | emb 4 | emb 5 |
| --- | --- | --- | --- | --- | --- | --- |
| multi | 1750 (selected 1500) | 41.06 | 37.32 | 31.69 | 32.48 | 39.32 |
| masked | 1000 (selected 750) | 75.88 | 75.96 | 75.52 | 74.19 | 73.54 |
| masked + div 0.5 | 750 | 75.89 | 72.06 | 70.05 | 67.41 | 66.93 |
| masked + div 2 | 750 | 74.06 | 72.17 | 70.39 | 65.98 | 64.46 |
| hps 0.5 | 1000 (selected 750) | 77.28 | 56.12 | 43.32 | 40.40 | 45.38 |
| hps 0.25 | 1000 | 74.63 | 42.03 | 43.76 | 41.57 | 47.47 |
| full sampling | 750 | 38.90 | 31.33 | 31.04 | 30.39 | 26.85 |
| saturate 25% | 500 | 36.94 | 32.01 | 30.87 | 28.83 | 35.74 |
| saturate 50% | 1000 | 35.68 | 40.27 | 37.13 | 36.40 | 36.36 |
| multi, LR 3e-5 | 1750 | 39.65 | 43.49 | 32.22 | 31.07 | 29.60 |

- **Runs with diverse queries (`multi`, sampling, LR 3e-5): every embedding is weak alone, and they are complementary.** Each finds 27–43% of a question's golds and scores 1.3–11.7 MRecall on its own; fused, they reach 38–43. For comparison, `single`'s one query finds 77.62% of the golds (53.30 MRecall).
- **`masked`: five copies of one good query.** Every embedding alone scores 46–49 and finds 74–76% of the golds; fusing them gives 48.59, no gain.
- **`hps`: one strong query plus weak extras.** Embedding 1 finds 75–77% of the golds (46.5–52.0 MRecall). Embeddings 2–5 find 40–56% (8–23 MRecall), and fusing them in lowers the score (`hps 0.5` step 1000: 51.98 alone, 47.27 fused).
- **Diversity penalty: the later embeddings get weaker, not complementary.** Embedding 1 is best (45–49), the rest decline (34–43), and fusion does not beat embedding 1 alone.
- No run has embeddings that are both strong alone and complementary. The diverse runs get the complementarity; the others get the strength.

## Findings by question

### Training objective

- `masked` (other golds are not negatives) gives the best multi-query dev and test, by collapsing its queries: cosine 0.94, flat k curve.
- The diversity penalty (relu of query–query cosine minus the matched golds' cosine) works as designed: q–q − g–g falls from +0.38 to +0.25 (λ = 0.5) and +0.02 (λ = 2). Test falls by 3–4.5 points. Spreading the queries does not make each one point at a different gold: the margin stays negative.
- `hps` weights 0.5 and 0.25 tie on dev (63.0 and 62.8). The single-query term is what makes `hps` work: its first query alone beats `single` on test (0.5: 54.05).

### Scheduled sampling

- Saturating the sampling rate early (25% or 50% of training) peaks around steps 500–1000 at 57–60 dev, below `multi`, then decays. By step 2500 the first query is useless (test k = 1: 0.75 and 0.38).
- Full sampling from the start (round 1) is worse: the first three queries are close to useless (test k = 1: 9.6).

### Learning rate

- LR 3e-5 is worse than 1e-5 for both modes: single 61.4 / 48.21 against 64.80 / 53.30; multi 55.8 / 39.92 against 61.2 / 38.42 (dev / test).

### Late training

Round-2 multi-query runs lose 6–23 dev points between their best checkpoint and step 2500; single-query loses 1.4. Keeping the final checkpoint (new this round) is what shows this.

| run | selected step | dev | test | test, k = 1 | g–g cos | final (2500): dev | test | test, k = 1 | g–g cos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| masked + div 0.5 | 750 | 61.4 | 43.13 | 48.59 | 0.58 | 52.8 | 39.55 | 29.19 | 0.58 |
| masked + div 2 | 750 | 60.0 | 44.63 | 45.01 | 0.61 | 54.0 | 38.04 | 29.76 | 0.60 |
| hps 0.25 | 1000 | 62.8 | 40.49 | 46.52 | 0.53 | 54.0 | 40.49 | 37.29 | 0.48 |
| saturate 25% | 500 | 57.2 | 41.24 | 7.53 | 0.53 | 34.6 | 21.28 | 0.75 | 0.42 |
| saturate 50% | 1000 | 60.4 | 43.31 | 6.59 | 0.52 | 44.8 | 23.92 | 0.38 | 0.43 |
| single, LR 3e-5 | 1250 | 61.4 | 48.21 | – | 0.61 | 60.0 | 46.52 | – | 0.61 |
| multi, LR 3e-5 | 1750 | 55.8 | 39.92 | 10.36 | 0.52 | 44.4 | 30.32 | 2.07 | 0.47 |

## Query and gold similarity

Computed on the full corpus for every evaluated checkpoint (`tools/geometry_eval.py`), test unless marked dev.

- **q–q cos**: mean pairwise cosine between a question's 5 query embeddings.
- **g–g cos**: mean pairwise cosine between a question's gold passages; **above random** subtracts the cosine of random passage pairs, so it compares across document spaces.
- **query→gold cos**: each query's cosine with the gold it is matched to (Hungarian); **margin**: that minus its cosine with the closest other gold of the question. A negative margin means a query is not specific to its gold.
- **sibling rank**: median number of corpus passages closer to a gold than its nearest sibling gold. Low means the golds sit together.

| run | step | q–q cos | g–g cos | q–q − g–g | g–g above random | query→gold cos | margin | sibling rank | dev q–q | dev g–g |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single | 2000 | – | 0.59 | – | 0.56 | 0.67 | +0.03 | 4 | – | 0.65 |
| multi | 1500 | 0.67 | 0.50 | +0.17 | 0.47 | 0.57 | +0.03 | 20 | 0.67 | 0.54 |
| masked | 750 | 0.94 | 0.57 | +0.38 | 0.54 | 0.55 | -0.04 | 3 | 0.95 | 0.64 |
| masked + div 0.5 | 750 | 0.84 | 0.58 | +0.25 | 0.56 | 0.55 | -0.04 | 4 | 0.84 | 0.67 |
| masked + div 2 | 750 | 0.63 | 0.61 | +0.02 | 0.59 | 0.51 | -0.03 | 3 | 0.65 | 0.68 |
| hps 0.5 | 750 | 0.64 | 0.53 | +0.11 | 0.50 | 0.48 | -0.01 | 5 | 0.65 | 0.56 |
| hps 0.25 | 1000 | 0.65 | 0.53 | +0.12 | 0.51 | 0.53 | +0.01 | 8 | 0.66 | 0.57 |
| full sampling | 750 | 0.60 | 0.52 | +0.09 | 0.50 | 0.61 | +0.05 | 10 | 0.58 | 0.58 |
| saturate 25% | 500 | 0.64 | 0.53 | +0.11 | 0.50 | 0.57 | +0.04 | 16 | 0.64 | 0.56 |
| saturate 50% | 1000 | 0.66 | 0.52 | +0.15 | 0.49 | 0.59 | +0.03 | 16 | 0.66 | 0.57 |
| single, LR 3e-5 | 1250 | – | 0.61 | – | 0.57 | 0.67 | +0.03 | 4 | – | 0.66 |
| multi, LR 3e-5 | 1750 | 0.67 | 0.52 | +0.15 | 0.48 | 0.59 | +0.04 | 16 | 0.66 | 0.55 |

**Untrained document space (reference).** Cell C keeps the base model's document encoder, so its g–g is the base retriever's: pending (C single full eval running). Comparing a run's g–g with it shows whether training pulled a question's golds together or apart.

*Figure (see the HTML page): Test MRecall@100 against gold–gold cosine. Large dots: selected checkpoints; small dots: final checkpoints (step 2500) of the round-2 runs. Filled: single-query; open: multi-query. Final checkpoints below 30 test are off the chart.*

- Single-query training keeps a question's golds together (g–g 0.59, sibling rank 4). Hungarian multi-query training separates them (0.50–0.53; sibling rank up to 16–20), presumably because each gold is a negative for the queries matched to its siblings.
- The masked loss removes that push (other golds are not negatives) and keeps g–g at single-query level (0.57–0.61, sibling rank 3–4). The diversity-penalty runs do too, yet score 3–4.5 points below `masked`: keeping the golds together is not enough on its own.
- The collapsing late checkpoints have the lowest g–g (0.42–0.47).
- The q–q cosine is about 0.6–0.67 for every multi-query run except `masked` and the penalised runs. Query similarity alone does not separate good runs from bad ones.

## Measurement

### The v1 test leak

44 of the 531 QAMPARI test questions had a near-duplicate in the v1 training data (gold-set Jaccard ≥ 0.8); v2/v3 remove them. The grid's best multi-query RRF score drops from 44.26 on all 531 to 41.48 on the 487 clean questions. On those, the current runs show no regression against the grid. Grid test numbers are not comparable with anything since.

| test MRecall@100, round-robin k = 5 | all 531 | 44 leaked | 487 clean |
| --- | --- | --- | --- |
| grid single (v1 data) | 52.73 | 84.09 | 49.90 |
| v3 single | 53.30 | 68.18 | 51.95 |
| grid multi (v1 data) | 38.23 | 63.64 | 35.93 |
| v3 multi, step 750 | 37.10 | 36.36 | 37.17 |
| v3 masked | 49.34 | 61.36 | 48.25 |

### Cheap dev and checkpoint selection

- Cheap dev scores each checkpoint on a reduced corpus (about 500k passages). Across runs it agrees with full dev (Spearman 0.91–0.96).
- Within a run it picks the full-dev-best checkpoint about half the time. Neighbouring checkpoints differ by about 1 point with a paired standard error of about 1 point, on full dev too.
- New protocol: full-evaluate the top 2 by cheap dev plus the final checkpoint; select by full dev; treat differences under 2 points as ties.

## In progress

| work | runs | status |
| --- | --- | --- |
| Wave 2: best hps weight combined with sampling | `hps + full sampling`, `hps + saturate 50%` | launches automatically once `hps 0.1` is evaluated (weight = best full dev of 0.5 / 0.25 / 0.1) and the disk projection allows |
| Wave 3: document encoder frozen or joint × query attention causal or bidirectional | C (frozen, bidirectional), A (frozen, causal), B (joint, causal); single and multi each. D = round 1. | C single: full evals; C multi: full evals; A single: cheap evals; A multi: waiting for GPUs; B single: waiting for GPUs; B multi: waiting for GPUs |

Early sign from wave 3: `C single` scores 37–39 on cheap dev at every step, against 71–74 for `single`. An earlier frozen-encoder run scored similarly (37.6, older dev set), and the code path checks out, so freezing the document encoder probably costs a lot. Cheap dev favours models whose document encoder was trained with hard negatives, so only the full evals compare fairly.

## Next steps and open questions

- **Multi-query as an add-on to a strong single query.** `hps`'s first query already beats `single`. Make the later queries add only what the first misses, for example by weighting the first query's list in the fusion or by stopping early when later queries add nothing.
- **Keep the golds together.** Gold–gold similarity follows quality. Test a multi-query loss that does not push sibling golds apart: masked loss without collapse, or a document-side term.
- **Second seed.** Many differences here are 1–3 points, about the size of seed and checkpoint noise. A second seed of the main runs would show which are real (on hold).
- **AmbigQA** is on hold until the QAMPARI picture settles.

## Appendix: setup and runs

- Model: inf-retriever-v1-1.5b, fine-tuned in full, 2 GPUs (FSDP), 2,500 steps, LR 1e-5 unless stated, one seed.
- Data: clean v3 QAMPARI with hard negatives (ratio 0.5); dev = clean v2 dev (500); test = the unchanged AMER test set (531).
- Corpus: Wikipedia chunks embedded at 512 tokens. Multi-query: k = number of golds in training, k = 5 at inference.
- Scheduled sampling: the rate of feeding back the model's own query embedding grows linearly to 1 at the end of training unless stated.

| run | group | change |
| --- | --- | --- |
| single | baseline | single-query baseline (LR 1e-5) |
| multi | baseline | multi-query baseline: Hungarian loss, scheduled sampling ramping to 1 at the end |
| masked | objective | masked Hungarian: the example's other golds are not negatives |
| masked + div 0.5 | objective | masked + diversity penalty, λ = 0.5 |
| masked + div 2 | objective | masked + diversity penalty, λ = 2 |
| hps 0.5 | objective | Hungarian + single-query loss on the first query, weight 0.5 |
| hps 0.25 | objective | Hungarian + single-query loss, weight 0.25 |
| hps 0.1 | objective | Hungarian + single-query loss, weight 0.1 |
| full sampling | sampling | sampling rate 1 from the start (always the model's own outputs) |
| saturate 25% | sampling | sampling rate reaches 1 at step 625 |
| saturate 50% | sampling | sampling rate reaches 1 at step 1250 |
| single, LR 3e-5 | learning rate | single-query, LR 3e-5 |
| multi, LR 3e-5 | learning rate | multi-query, LR 3e-5 |
| hps + full sampling | combination (wave 2) | best hps weight + sampling rate 1 from the start |
| hps + saturate 50% | combination (wave 2) | best hps weight + rate reaching 1 at step 1250 |
| C single | doc encoder × attention (wave 3) | frozen doc encoder, bidirectional |
| C multi | doc encoder × attention (wave 3) | frozen doc encoder, bidirectional |
| A single | doc encoder × attention (wave 3) | frozen doc encoder, causal query attention |
| A multi | doc encoder × attention (wave 3) | frozen doc encoder, causal query attention |
| B single | doc encoder × attention (wave 3) | joint doc encoder, causal query attention |
| B multi | doc encoder × attention (wave 3) | joint doc encoder, causal query attention |

Detailed reports: `reports/ablation_clean_v3_2026-10-07.md` (round 1, leak), `reports/cheap_eval_validity_2026-10-07.md`, `reports/wave3_prep_2026-10-09.md`. Data behind this page: `reports/weekly_summary_data.json`, built by `training/inf_retriever/tools/weekly_collect.py`.

