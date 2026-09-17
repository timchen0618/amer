# Debugging Plan: Multi-Query AR Retrieval (EmbeddingModelDocEncNoProj)

## Symptoms

- Finetuned single-query baseline: mRecall@k = 20%
- Finetuned multi-query model: mRecall@k = 0%
- Eval accuracy and MRR during training remain high throughout

---

## Root Cause Analysis

### 1. The eval function never tests actual multi-query generation

`evaluate()` in `finetuning_multi.py` runs a single encoder forward pass and checks if
the one resulting embedding retrieves any of the nqe positives. It never calls
`model.generate()`. With nqe=5 positives in the candidate pool, this metric is easy to
satisfy and says nothing about whether the model can generate 5 diverse embeddings.
High eval acc/MRR during training is therefore misleading.

### 2. The model converges under near-pure teacher forcing

The sampling schedule is `sampling_rate = step / total_steps`. With ~2100 total steps
and loss converging at ~200 steps, the model converges at sampling_rate ≈ 0.095 (9.5%
self-sampled). It has found a solution to the teacher-forced problem before ever
meaningfully training on its own outputs. At inference, `sampling_rate = 1.0` — the
model sees a completely different input distribution and produces garbage, hence 0%
mRecall.

### 3. Potential mode collapse

`HungarianMaskedContrastiveLoss` masks unassigned same-example positives from the
denominator. A model that places all k embeddings near the centroid of the k golds can
achieve low loss because each embedding is assigned to a different gold (Hungarian
matching is flexible) and the other unassigned positives are hidden. The result: k
near-identical embeddings that all retrieve the same documents, giving no benefit over
single-query retrieval.

Note: Hungarian matching does provide a gradient signal for diversity — if all k
embeddings are at the centroid, they get assigned to different golds and pulled in
different directions. But this signal is too weak if the model has already converged
under teacher forcing.

---

## Diagnostics Added

The following metrics are now logged to wandb. Run a training job and inspect these
before applying fixes.

### During training (`EmbeddingModelDocEncNoProj.forward()`)

| Metric | What to look for |
|--------|-----------------|
| `train/sampling_rate` | Confirms the 0→1 ramp; check value at the step where loss converges |
| `train/step_j_teacher_cos_sim` | Cosine sim between step j output and its teacher (gold) embedding; if this drops sharply as sampling_rate increases later in training, confirms the model is only calibrated for teacher-forced inputs |
| `train/pairwise_cos_sim` | Avg off-diagonal cosine sim among k generated embeddings; rising toward 1.0 = mode collapse |

### During evaluation (`evaluate()`)

| Metric | What to look for |
|--------|-----------------|
| `eval_step_j_acc` | Per-step in-batch retrieval accuracy using AR-generated embeddings (no teacher); if step 0 is decent but steps 1–4 crash to near 0, confirms train/inference mismatch |
| `eval_pairwise_cos_sim` | Same pairwise sim at inference time; near 1.0 = mode collapse |
| `eval_same_top1_pct` | % of queries where all k embeddings retrieve the same top-1 doc (strict collapse check) |
| `eval_repeat_pct` | `(k - num_unique_top1_docs) / (k - 1)` per query, averaged; 0% = all steps retrieve different docs, 100% = all steps retrieve the same doc |

### How to interpret results

- **`eval_step_0_acc` decent, steps 1–4 near 0** → train/inference mismatch is primary cause; fix the schedule (Fix 1 below)
- **`eval_pairwise_cos_sim` and `eval_repeat_pct` near 100%** → mode collapse; add diversity loss (Fix 2 below)
- **Both** → apply Fix 1 first, then reassess

---

## Fixes

### Fix 1: Sampling schedule (highest priority)

**Problem:** Model converges at sampling_rate ≈ 0.095, never learns inference-time behavior.

Change the `sampling_rate` argument passed in the training loop
(`finetuning_multi.py`, the `model(**batch, sampling_rate=...)` call):

```python
# Current (broken): starts at 0, model converges under teacher forcing
sampling_rate = step / opt.total_steps

# Option A — no teacher forcing at all (recommended starting point)
sampling_rate = 1.0

# Option B — start at 0.5, ramp to 1.0
sampling_rate = 0.5 + 0.5 * (step / opt.total_steps)
```

Option A is the cleanest: training and inference are identical, no mismatch possible.
The Hungarian loss still drives each embedding toward a different gold.
Option B gives some stability early in training.

Try Option A first. If training is unstable (loss spikes and does not recover), fall
back to Option B.

### Fix 2: Explicit diversity loss

**Problem:** Hungarian matching provides diversity through gradient pressure but it is
indirect. An explicit repulsion term is stronger and more direct.

Add to `EmbeddingModelDocEncNoProj.forward()` after `selected_outputs_embeddings` is
computed:

```python
outputs_norm = F.normalize(selected_outputs_embeddings, dim=-1)  # (bsz, k, d)
sim_matrix = torch.bmm(outputs_norm, outputs_norm.transpose(1, 2))  # (bsz, k, k)
eye = torch.eye(output_len, device=sim_matrix.device).unsqueeze(0)
diversity_loss = sim_matrix.masked_fill(eye.bool(), 0.0).sum() / (bsz * output_len * (output_len - 1))
loss = contrastive_loss + lambda_div * diversity_loss
```

This penalizes pairwise cosine similarity among the k embeddings directly. Combined
with Hungarian matching, you get push-pull dynamics: contrastive loss pulls each
embedding toward its assigned gold, diversity loss pushes embeddings away from each
other.

Start with `lambda_div = 0.1` and tune. Watch `train/pairwise_cos_sim` — it should
decrease once this term is active.

### Fix 3: Harder negatives

**Problem:** Random negatives are too easy. The model saturates the contrastive loss
at step ~200 before scheduled sampling kicks in, because distinguishing a random
negative from the correct gold is trivial.

Set `--negative_hard_ratio 0.5` (or higher) to mix in hard negatives — documents that
are topically related but not answers. This keeps the problem hard longer, preventing
early convergence and giving the sampling schedule time to matter.

This is complementary to Fix 1 and 2, not a standalone solution.

---

## Recommended Order of Experiments

1. **Run diagnostics first** (already added). Confirm whether mismatch, collapse, or
   both are present by inspecting `eval_step_j_acc`, `eval_repeat_pct`, and
   `eval_pairwise_cos_sim`.

2. **Apply Fix 1** (`sampling_rate = 1.0`). This is the single most likely cause of
   0% mRecall. Re-run and check if mRecall recovers.

3. **If collapse persists** (high `eval_repeat_pct` after Fix 1), **apply Fix 2**
   (diversity loss with `lambda_div = 0.1`).

4. **If training is still unstable or converges too fast**, **apply Fix 3** (harder
   negatives).

---

## What Good Results Should Look Like

After fixes:
- `eval_step_j_acc` should be similar across all j steps (not degrading)
- `eval_repeat_pct` should be well below 50%
- `eval_pairwise_cos_sim` should be below 0.5
- mRecall@k should exceed the single-query finetuned baseline (20%)
