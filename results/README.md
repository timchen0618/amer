# results/

Mostly gitignored (retrieval outputs, logs). The files below are the tracked, canonical ones.

## QAMPARI S0 split, frozen doc encoder (session F, `training/inf_retriever/finetuning_multi_paper.py`)

| File | What it holds |
|---|---|
| `phaseC/seed_report.md` / `.json` | **Headline table.** Single, cap, no-cap and no-cap io_proj, seeds 0-2, at the dev-selected step and at step 10000: dev, test, large / small test thirds, mean ± sd, gap to single. Built by `training/inf_retriever/tools/seed_report.py` |
| `phaseC/fusion_S0_test.json`, `fusion_S0_rseed_test.json` | Multi runs, seed 0 and seeds 1-2: every fusion method on full dev, test and the thirds (`rrf_c60` = headline) |
| `phaseC/fusion_S0_single_test.json`, `fusion_S0_single_rseed_test.json` | The same for single-query runs (key `single`) |
| `phaseC/geometry_ref/compare.md`, `target_coverage/table.md` (+ `.json`) | Embedding-space geometry and per-target coverage, all three seeds |
| `phaseC/geometry_seed_summary_sel.json` | Per run and seed, at the dev-selected step: geometry stats on all / large / small test questions |
| `phaseC/fusion/*.json` | Session N's fusion scores (Nemotron `n_*`, Qwen3 `q_*`, legacy one-label step `olshuf_*`) |
| `phaseC/full/README_slim.md` | What slimming keeps in `phaseC/full/` retrieval outputs |

Untracked but needed: `phaseC/clean_v3_f_qampari_<run>.driver.out` (driver logs; `seed_report.py` reads the dev rankings from them) and `phaseC/full/qampari/<run>_s<step>/` (per-step lists and query embeddings).

## Paper reproduction
Outputs are in `paper_repro_data/` (not in git); report in `reports/paper_repro_2026-10-23/`.
