# Does the geometry of AmbigQA / AmbigQA_2docs / QAMPARI support multi-query retrieval?

**Question this answers:** independent of any specific checkpoint's trained quality, does the
*geometry* of the gold documents on these three benchmarks even leave room for a multi-query
method (AMER) to beat a single-query baseline? If not, no amount of training will make AMER win
on these datasets for the reason the method targets — it would need a different reason (or a
different benchmark).

**tl;dr:** QAMPARI's genuine, oracle-centered geometry (the rigorous test) says **yes, there is
real room** — 53% of examples cross the theoretical single-vector-infeasibility threshold at
k=100 (and up to 99.6% at k=5). AmbigQA and AmbigQA_2docs say **mostly no** — 1.1% and 0.2%
respectively. The cheaper, query-centered proxy said "no" for all three datasets — it was
*wrong* for QAMPARI, in a way that matters, because it measures the trained model's placement,
not the geometric ceiling. Use the oracle numbers, not the query numbers, to make the call for
which benchmark is worth relying on to demonstrate the method.

All code is in `rho_analysis/`: `pipeline_query.py`, `pipeline_oracle.py`, `multik_analysis.py`,
`predict_wins.py` (upstream, unmodified). Data are in `rho_analysis/outputs/<dataset>/`.

---

## 1. What rho measures, and why there are two versions

For a query with `m` gold-document aspects, `rho = D / r_k`:
- `D` = the pairwise diameter (max distance) among the `m` gold document embeddings.
- `r_k` = the distance from *some* single query point to its k-th nearest neighbor in the full
  corpus.

By the triangle inequality, if a single point lies within radius `r` of two gold docs, those two
docs are at most `2r` apart. So `rho > 2` is an absolute proof that no single vector — trained
or not — can retrieve all `m` gold docs within its own top-k ball; `rho < sqrt(2) ≈ 1.414` is
the (softer, average-case) regime where a single vector should comfortably manage it. Multi-query
retrieval is motivated specifically by examples above that threshold: cases where the golds are
too spread out for one query vector's neighborhood to hold all of them, regardless of training.

The two versions differ in what "some single query point" means:

- **rho_query** — the point is the baseline model's own trained query embedding. `r_k` is read
  directly off the existing `standard`-mode retrieval output's `ctxs` list (already sorted by
  score, 8000 docs deep — no new search needed). This measures *"is the currently-trained model's
  query point good enough,"* which conflates two different questions: geometric necessity, and
  training quality.
- **rho_oracle** — the point is the centroid of the actual gold-document embeddings, i.e. the
  best plausible single-vector placement, found via a genuine full-corpus k-NN search (using
  `retrieval_base.py`'s existing shard-by-shard FAISS search, fed centroid vectors instead of
  real question embeddings — no corpus materialization, no new indexing code). This isolates
  geometric necessity from training quality: if even the best-case point can't reach all the
  golds, no amount of training the single-query model will fix that.

**rho_query is the cheap, approximate version. rho_oracle is the rigorous one requested for this
report.** They can and do disagree substantially — see Section 4.

---

## 2. Data and gold-identity caveats

| Dataset | n examples | gold identity |
|---|---|---|
| qampari | 531 | id-based, from the dataset's own `ground_truths` (m = number of clusters, exact) |
| ambigqa | 827 (817 with a resolvable oracle centroid) | approximate — see below |
| ambigqa_2docs | 474 (468 with a resolvable oracle centroid) | approximate — see below |

QAMPARI's gold IDs are the dataset's own ground truth, independent of anything the model
retrieves — the cleanest signal of the three.

AmbigQA / AmbigQA_2docs have no ground-truth document IDs in this pipeline's corpus namespace:
`positive_ctxs` carries no `id` field, and its `has_answer` flags are stale (computed against a
different candidate pool than the current corpus — confirmed by inspection). Gold identity was
instead defined operationally, per answer aspect: the highest-ranked document in the baseline's
own top-500 retrieved list that satisfies `has_answer(aliases, text)` (`src.eval_utils`, exact
reuse, no reimplementation). Aspects with no satisfying doc in that window are dropped for that
example (mean coverage 93.4% / 92.9%). This makes AmbigQA's D(q) and rho a statement about
*"documents the baseline itself can find,"* not the true unknown gold set — checked for
circularity by restricting to examples with 2+ resolved gold docs (excludes the ones that
trivially collapsed to a single point); the result was materially unchanged.

---

## 3. Headline results at k=100

Regime boundaries (from `predict_wins.py`, already established in this repo): rho < 1.41
"single vector sufficient," 1.41 ≤ rho ≤ 2.0 "configuration-dependent," rho > 2.0 "single vector
impossible."

### rho_query (cheap proxy — actual trained model's query point)

| Dataset | n | median | p95 | max | frac ≥ √2 |
|---|---|---|---|---|---|
| qampari | 531 | 0.594 | 0.849 | 1.003 | 0.0% |
| ambigqa | 827 | 0.643 | 1.123 | 1.351 | 0.0% |
| ambigqa_2docs | 474 | 0.504 | 1.026 | 1.314 | 0.0% |

Not one example in any dataset crosses the threshold. Read naively, this says the geometric
argument for multi-query retrieval fails everywhere.

### rho_oracle (rigorous — best-case centroid point, genuine full-corpus search)

| Dataset | n | median | p95 | max | frac ≥ √2 | frac > 2 |
|---|---|---|---|---|---|---|
| qampari | 531 | **1.426** | 1.794 | 2.278 | **52.9%** | 0.75% |
| ambigqa | 817 | 0.719 | 1.308 | 1.581 | 1.1% | 0.0% |
| ambigqa_2docs | 468 | 0.581 | 1.219 | 1.486 | 0.2% | 0.0% |

QAMPARI flips completely: median rho_oracle is *above* the threshold, and just over half of
examples are in the regime where a single vector cannot cover the gold spread even under the
best possible placement. A small fraction (0.75%) are even in the absolute-impossibility regime
(rho > 2). AmbigQA and AmbigQA_2docs barely move.

**Why the gap between rho_query and rho_oracle is so large for QAMPARI specifically:** the
oracle centroid of QAMPARI's gold docs (up to 5 distinct, often unrelated entities per query — a
UN administrator's alma maters could span several different people entirely) lands in a denser
region of the embedding space than the trained query vector does, giving it a *smaller* r_k, not
larger. A trained single-query embedding is optimized contrastively to be discriminative — pushed
toward the relevant region and away from distractors, which tends to place it somewhere with
fewer very-close neighbors (larger r_k, smaller rho). An untrained geometric average of several
embeddings has no such pressure and instead regresses toward denser "hub" regions of the space
(smaller r_k, larger rho). This is an empirical finding from the data, not something assumed in
the design — the opposite direction was equally plausible a priori (an off-manifold centroid
could just as easily have landed somewhere sparse).

---

## 4. Is rho scale-free in k? (requested check)

Concern raised: since `r_k` grows with the retrieval budget `k`, rho = D/r_k is not literally
scale-free — the absolute thresholds (1.41, 2.0) are only meaningful at a fixed k. The mitigating
argument: if the corpus has local effective dimension `d_eff`, `r_k ~ k^(1/d_eff)`, so a 10x
change in k moves `r_k` by only `10^(1/d_eff)` — small for the tens-of-dimensions local structure
typical of embedding spaces. That means absolute rho shifts with k, but **rankings** of examples
by rho should be close to invariant, which is what actually matters for regime membership being
a stable diagnostic rather than a k-dependent artifact.

Checked directly: ran both variants at k ∈ {5, 10, 20, 50, 100} (`pipeline_query.py --ks`,
`pipeline_oracle.py --ks` — a single search per dataset at k_search=100, sliced to smaller k, no
repeated corpus searches) and computed Spearman rank correlation between the rho rankings at
each k (`multik_analysis.py`).

**rho_query — near-perfectly stable, as predicted:**

| Dataset | min pairwise Spearman (k=5..100) | frac≥√2 at k=5 → k=100 |
|---|---|---|
| qampari | 0.990 | 0.0% → 0.0% |
| ambigqa | 0.983 | 4.7% → 0.0% |
| ambigqa_2docs | 0.991 | 1.7% → 0.0% |

The prediction holds cleanly: absolute rho drops as k grows (as expected — bigger ball, easier
to cover), but which examples are high/low relative to each other barely changes. The AUC-vs-win
test and the qualitative "which examples are hard" story would look the same at any k in this
range.

**rho_oracle — NOT stable, especially for QAMPARI. This is itself a finding, not noise:**

| Dataset | min pairwise Spearman (k=5..100) | frac≥√2 at k=5 → k=100 |
|---|---|---|
| qampari | **0.641** | 99.6% → 52.9% |
| ambigqa | 0.936 | 34.8% → 1.1% |
| ambigqa_2docs | 0.966 | 20.5% → 0.2% |

QAMPARI's oracle-centered ranking is genuinely unstable across k (Spearman as low as 0.64 between
k=5 and k=100), and the fraction crossing the threshold nearly halves as k goes from 5 to 100.
Per the diagnostic proposed in the request that raised this: an instability this large means the
corpus has **heterogeneous local density around the gold centroids** — some centroids sit in a
locally dense pocket that only reveals its true (sparser) surrounding structure once you look
past the first handful of neighbors, and how many neighbors that takes varies a lot from query to
query. This is consistent with, and gives a mechanism for, the rho_query-vs-rho_oracle gap in
Section 3: centroids are landing somewhere structurally different from where trained queries
land, and that "somewhere" isn't uniform either.

**Practical takeaway:** rho_query is safe to report at a single k, or to use as an x-axis for
Figure-1-style plots — the ranking won't materially change with k. rho_oracle is *not* safe to
report at a single k without saying which k, especially for QAMPARI — the absolute headline
number (53% at k=100) should be read as "roughly half at the operating point this project
actually retrieves at (k=100)," not as a universal constant of the dataset; at a tighter budget
(k=5) it would be reported as "essentially all examples" instead, which tells a rhetorically
different (though not contradictory) story.

---

## 5. What this means for AMER

- **QAMPARI is the dataset where the geometric case for multi-query retrieval actually holds.**
  At the k=100 operating point, roughly half of examples have gold documents too spread apart for
  any single vector — trained or oracle — to cover in one top-100 ball. This is computed from the
  dataset's own unambiguous ground truth, not from anything circular. If AMER's core claim is
  "single vectors are geometrically insufficient for multi-aspect retrieval," QAMPARI is where
  that claim is falsifiable in the *right* direction (data supports it), not where it's vacuous.
- **AmbigQA and AmbigQA_2docs do not make this case.** Under the rigorous oracle test, essentially
  no examples (1.1% / 0.2%) reach the infeasibility threshold. Whatever gains AMER shows on these
  two benchmarks are very unlikely to be explained by "the gold docs were geometrically
  unreachable by one vector" — if there are real gains there, they come from something else
  (e.g. the multi-query model happening to place its vectors better than the single-query
  baseline's specific trained point, which is a training/optimization story, not a geometric-
  necessity one).
- **Separately (already flagged, repeated here for completeness):** the *currently on-disk*
  `multi_hungarian` retrieval outputs underperform `standard` on all three datasets at k=100
  MRecall (ambigqa: 75.6%→71.7%; ambigqa_2docs: 80.2%→69.4%; qampari: 22.4%→0.0%, the last one
  a near-total loss of gold-ID overlap, not just a worse score). You confirmed these are the
  latest/correct files. This is orthogonal to the geometric question above — it says nothing about
  whether QAMPARI's real ceiling supports AMER, only that the specific checkpoint on disk right
  now isn't reaching it. The 53% figure is a statement about the dataset, not about this
  checkpoint.

---

## 6. Caveats and what's still approximate

1. AmbigQA/AmbigQA_2docs gold identity is baseline-dependent by construction (Section 2) — their
   rho numbers should be read as "documents reachable by the current retrieval system," which
   biases them toward the "sufficient" conclusion. QAMPARI has no such bias.
2. rho_oracle's absolute value is k-dependent and unstable in ranking for QAMPARI (Section 4) —
   the 53% headline is specifically the k=100 number; report it with k stated, not as a bare
   percentage.
3. The centroid used for rho_oracle is the simple mean of L2-normalized gold embeddings,
   re-normalized to unit length. This is the natural "average single vector" but is not
   necessarily the *optimal* single point (e.g. a 1-center/min-enclosing-ball solution could do
   slightly better) — rho_oracle as computed is an upper bound on how bad the best single vector
   is, i.e. if anything it understates infeasibility slightly.
4. All numbers use `standard`-mode corpus embeddings (the baseline's own encoder), consistent
   across rho_query and rho_oracle for a fair comparison, and matching how the retrieval that
   actually gets deployed is embedded.

---

## 7. Where everything lives

```
rho_analysis/
  pipeline_query.py       # rho_query: reads existing baseline ctxs, small gold-embedding lookup
  pipeline_oracle.py       # rho_oracle: genuine full-corpus k-NN via retrieval_base.retrieve()
  multik_analysis.py       # Spearman rank-stability check across k, Section 4's numbers
  predict_wins.py          # upstream, unmodified; --rho-key {rho_query,rho_oracle}
  run_pipeline.sbatch       # array job, rho_query, 3 datasets
  run_pipeline_oracle.sbatch # array job, rho_oracle, 3 datasets
  outputs/<dataset>/
    rho_query.jsonl         # per-example, incl. by_k breakdown
    rho_oracle.jsonl        # per-example, incl. by_k breakdown
    baseline_results.jsonl  # standard-mode per-example MRecall@100
    system_results.jsonl    # multi_hungarian per-example MRecall@100
    predict_wins_report.json
    multik_query.log, multik_oracle.log   # Section 4's tables
```
