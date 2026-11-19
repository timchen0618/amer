# Session N: F's frozen QAMPARI S0 setting with Nemotron-3.5-Embed-1B (2026-11-13)

Rerun of session F's frozen-doc-encoder QAMPARI setting (S0 data, 10k steps, lr 1e-5, temperature 0.05, one-gold step)
with `nvidia/Nemotron-3.5-Embed-1B-BF16` in place of `infly/inf-retriever-v1-1.5b` (handoff:
`F_retriever_swap_handoff_2026-11-12.md`). Bidirectional attention + mean pooling (Nemotron's native setup).
Selection as in F: test at the dev-selected step (full S0 dev, 2459 questions); a difference under 2 points is a tie.
MRecall@100; multi = RRF over k = 5 embeddings. Thirds = the paper's large / small distance thirds of test
(168 / 180 questions, `tools/fusion_eval.py`, `results/phaseC/fusion/n_nemotron.json`).

## Results (dev-selected step)

| setting | retriever | step | dev | test | large / small |
|---|---|---|---|---|---|
| base (no training) | inf-retriever | - | - | 24.48 | |
| base (no training) | Nemotron | - | 11.79 | 18.64 | |
| single, 2 updates / step | inf (F `S0_single_10k_2step`) | 10000 | 51.93 | 35.03 | 25.00 / 50.00 |
| single, 2 updates / step | **Nemotron** | 10000 | 54.09 | 34.84 | 26.19 / 50.00 |
| multi, linear 0->1 + one-gold | inf (new, `clean_v3_n_qampari_S0_10k_onelabel`) | 9000 | 50.83 | 33.90 | 25.00 / 45.56 |
| multi, linear 0->1 + one-gold | **Nemotron** | 6000 | 49.45 | 31.07 | 21.43 / 45.56 |
| multi, cap 0.8 / x10 + one-gold | inf (F `S0_cap10k_onelabel`) | 10000 | 50.71 | 33.15 | 25.00 / 43.89 |
| multi, cap 0.8 / x10 + one-gold | **Nemotron** | 5000 | 46.28 | 28.63 | 20.24 / 44.44 |

Other tested steps: Nemotron single s9000 54.05 / 35.03, s4000 50.39 / 32.39; Nemotron linear s7000 48.76 / 30.13,
s9000 47.42 / 29.38, s10000 46.40 / 29.57; Nemotron cap s4000 45.99 / 28.63, s10000 44.37 / 28.25; inf linear
s8000 50.67 / 33.52, s10000 50.47 / 32.96.

- **Single:** tie on test (-0.2), Nemotron +2.2 on dev, large/small equal.
- **Multi, linear:** Nemotron -2.8 on test (not a tie); dev tie (-1.4). Nemotron dev peaks at 6000 and then falls
  (49.45 -> 46.40 at 10000); inf keeps rising to 9000.
- **Multi, cap:** Nemotron -4.5 on test, -4.4 on dev. With inf the cap and linear schedules tie; with Nemotron cap is
  worse than linear (-2.4 test).
- So the multi-vs-single gap widens with Nemotron: inf multi is 1.1 (linear) / 1.9 (cap) below inf single on test;
  Nemotron multi is 3.8 / 6.2 below Nemotron single.

## Where the multi loss comes from (test, RRF over the first k embeddings)

| run | k=1 | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|---|
| inf linear s9000 | 34.09 | 29.94 | 34.09 | 33.52 | 33.90 |
| Nemotron linear s6000 | 33.90 | 25.24 | 29.00 | 28.81 | 31.07 |
| inf cap s10000 (F) | 34.65 | 32.77 | 31.26 | 33.33 | 33.15 |
| Nemotron cap s5000 | 32.20 | 25.05 | 25.80 | 25.05 | 28.63 |

The first embedding alone is about as good as with inf (and as Nemotron single); adding the later embeddings costs
much more with Nemotron. Geometry (test, `*_geom.json`): Nemotron's doc space is far less anisotropic (mean pairwise
cosine of a question's gold passages 0.41 vs 0.86 with inf), and the generated queries spread more (pairwise query
cosine 0.65 linear / 0.56 cap vs 0.77 / 0.70 with inf). Absolute cosines are not comparable across the two spaces,
but the later queries clearly land on worse lists, and RRF with them drags the first list down.

## Caveats

- The S0 hard negatives (and the S0 split) were mined with inf-retriever and reused unchanged; Nemotron's own hard
  negatives would differ. Base Nemotron is 5.8 points below base inf-retriever on QAMPARI test.
- lr 1e-5 kept from F for comparability. Nemotron moves fast: after 30 steps (warmup) query embeddings have cosine
  0.74 to the base and dev is already ~30; the multi runs peak early (5000-6000) and then decline, which suggests the
  lr / schedule is not tuned for Nemotron.
- Nemotron's 2048-d embeddings are used directly (no projection layer), like inf's 1536-d.
- `perstep_eval.py` fails on every S0 run ("per-embedding scoring failed"; it hardcodes the old 500-question dev),
  also for F's runs; it only drops an extra diagnostic.

## Implementation (worktree `.claude/worktrees/n-nemotron`, branch `nemotron-retriever`, not merged)

- `--retriever {inf,nemotron}` (default inf) in `finetuning_multi_nemo.py`; `src/nemotron.py` (format
  `query: ` / `passage: `, no BOS/EOS, bf16, mean pooling, bidirectional text backbone); F's scripts' defaults unchanged.
- Environment `/scratch/hc3337/envs/nemotron.ext3` (div clone + transformers 5.20, torch 2.9 cu128; offline hub);
  `div.ext3` untouched.
- Corpus: `wikipedia_embeddings/qampari/nemotron_frozen_base_512` (32 shards, 25.9M passages, 2048-d fp16, ~106 GB).
- Retrieval is two-stage (queries embedded in nemotron.ext3, faiss search in nli), since nemotron.ext3's faiss is
  CPU-only.
- Launch: `bash training/inf_retriever/tools/n_launch.sh <suffix> <inf|nemotron> "<flags>"` (worktree).

## Round 2: closing the multi-vs-single gap (2026-11-14, partial)

Gap = multi large third - single large third (test, dev-selected step; `results/phaseC/fusion/n_round2.json`).
Round 1 (lr 1e-5, inf-mined HN, no projection): single 34.84 (26.19 / 50.00); linear 31.07 (21.43 / 45.56), gap -4.8;
cap 28.63 (20.24 / 44.44), gap -6.0. inf reference: gap 0 / 0.

| intervention | single: step, test (large / small) | linear: step, test (large / small), gap | cap: step, test (large / small), gap |
|---|---|---|---|
| lr 5e-6 | s9000, 32.58 (23.21 / 48.33) | s7000, 26.74 (17.86 / 40.56), **-5.4** | s9000, 25.80 (15.48 / 40.56), **-7.7** |
| Nemotron-mined HN | s8000, 35.40 (25.60 / 49.44) | s7000, 31.83 (22.02 / 44.44), **-3.6** | s5000, 28.44 (20.24 / 40.56), **-5.4** |
| io_proj (single: output only) | s8000, 34.84 (24.40 / 51.67) | s7000, 31.26 (23.81 / 44.44), **-0.6** | s3000, 28.25 (19.64 / 42.22), **-4.8** |

RRF test by number of embeddings (k = 1..5): io_proj linear 34.46 / 28.44 / 28.06 / 29.76 / 31.26; HN linear
32.39 / 26.55 / 28.81 / 29.76 / 31.83; lr 5e-6 linear 30.51 / 23.16 / 23.16 / 23.92 / 26.74 (round 1 linear:
33.90 / 25.24 / 29.00 / 28.81 / 31.07).

- lr 5e-6 hurts single and multi alike (single -2.3 test) and widens the gap: dropped.
- Nemotron's own hard negatives: single +0.6 test, linear +0.8 test, gap narrows by 1.2 (under 2 points: a tie).
- io_proj: overall test unchanged (single 34.84 = round 1; linear 31.26 vs 31.07), but the large-third gap shrinks to
  -0.6: the multi large third rises 21.43 -> 23.81 and the single large third falls 26.19 -> 24.40. Against the round-1
  single it is -2.4. Thirds have 168 questions (one question = 0.6 points), so read this with care. The first
  embedding alone (k=1, 34.46) is the best io_proj multi list; the later embeddings still pull RRF down.
