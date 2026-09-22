# Retrieval geometry analysis

Diagnostics for the AMER paper. Answers one question: **is the multi-vector query
encoder solving a problem these benchmarks actually contain?**

## Why

Appendix A.4 reports same-query golds at cosine distance 0.10 (AmbigQA) vs 1.05 in
the synthetic data, read as "the benchmark isn't diverse enough." Two problems:

1. **Those numbers aren't comparable.** Synthetic vectors are isotropic; real encoder
   spaces occupy a narrow cone, which compresses every cosine distance toward zero.
   0.10 might be a large gap in a squashed space.
2. **Raw distance is the wrong x-axis.** Whether one query vector can cover all golds
   depends on gold-set diameter *relative to local corpus density*. Golds 0.3 apart are
   trivial in a sparse region, impossible in a dense one.

So define

```
rho = D(q) / r_k(q)        D = max pairwise distance among golds
                           r_k = distance to the k-th nearest corpus doc
```

A set of diameter `D` needs an enclosing ball of radius `>= D/2`, so `rho <= 2` is
**necessary** for single-vector coverage at rank k; Jung's theorem gives `rho <= 1.41`
as **sufficient** in high dimensions.

| regime | meaning |
|---|---|
| `rho < 1.41` | one vector suffices in principle — multi-vector should not help |
| `1.41–2.0` | configuration-dependent |
| `rho > 2.0` | no single vector can do it — the regime AMER is for |

If almost nothing sits above 2, your gains come from better centroid placement, not
cluster coverage, and the paper's stated mechanism isn't what's happening.

## Files

| file | step |
|---|---|
| `space_shape.py` | participation ratio / effective rank / cone collapse — how squashed is the index |
| `rho.py` | per-example `D`, `r_k`, `r_{k/m}`, `rho` |
| `whitening.py` | rescale principal directions, both sides, then recompute |
| `predict_wins.py` | does `rho` predict AMER wins? AUC vs. raw distance, absolute diffs with bootstrap CIs |
| `run_all.py` | driver, raw + whitened, prints the decision |
| `make_demo_data.py` | synthetic smoke test |
| `io_utils.py` | loading, kNN radii (faiss if present, numpy fallback) |

## Input contract

```
data_dir/
  corpus_emb.npy          (N, d) float32 — the document index you already built
  q_single.npy            (Q, d) single-query baseline vectors
  qids.json               ["q0", "q1", ...] aligning q_single.npy rows
  golds.jsonl             {"qid": "q0", "gold_ids": [1423, 88201, ...]}   row indices into corpus_emb
  baseline_results.jsonl  {"qid": "q0", "mrecall": 1}   per-example, not aggregated
  amer_results.jsonl      {"qid": "q0", "mrecall": 0}
```

Everything is L2-normalized internally; distance is Euclidean on the unit sphere
(monotone in cosine, but obeys the triangle inequality, which the covering argument needs).

## Usage

```bash
pip install numpy          # faiss-gpu optional, ~100x faster on 25M docs
python make_demo_data.py --out-dir demo
python run_all.py --data-dir demo --k 100 --m 3 --no-faiss   # smoke test

python run_all.py --data-dir /path/to/qampari --k 100 --m 5  # real run
```

Individual steps take the same flags; see each file's docstring.

## Reading the output

`run_all.py` prints median `rho` and `frac(rho > 2)` for the raw and whitened space.

- **small both** → benchmark can't show what AMER is for. Reframe around benchmark
  construction; look for a high-rho dataset (NERetrieve is a candidate).
- **meaningful mass above 2** → you have a per-example predictor of single-vector
  failure. Report the AUC, make the criterion the contribution. This is the analysis-paper
  version and it survives unimpressive absolute numbers.
- **small raw, large whitened** → golds *are* separable but the encoder's geometry hides
  it. The constraint is document-side, so every query-side method (expansion,
  decomposition, MMR, MMLF, POQD, AMER) is pushing the wrong lever. Train the
  cached-embedding document adapter and show gains growing with `rho`.

## Two things this replaces in the draft

- **Figure 1's x-axis.** Swap raw mean pairwise distance for `rho`. `predict_wins.py`
  reports both AUCs so you can show the improvement rather than assert it.
- **Figure 6.** "Are same-query golds farther apart than golds from two random queries"
  is nearly guaranteed to come out the way it did and doesn't bear on retrievability.
  "Is the gold set wider than one retrieval ball" is the question that does.

`rho.py` also reports `budget_shrink` = `r_k / r_{k/m}`: the radius each of m vectors
loses when they split a top-k budget round-robin. That's the cost side of going
multi-vector, and it quantifies the "diversity can hurt when targets are similar"
effect at L368, which is currently explained only by hand.

## Caveats

- `rho` uses the same encoder that produced the index, so it inherits that encoder's
  biases. It's a statement about retrievability under *this* geometry, not about semantics.
- `rho_oracle` centers `r_k` on the gold centroid (best case for any single vector);
  `rho_query` centers it on your actual baseline query. A large gap means your model is
  misplaced rather than fundamentally limited — a different diagnosis.
- Gold sets from string matching contain false positives, which inflate `D`. Run the
  label audit before trusting the high-`rho` tail.
