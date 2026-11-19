# Clean v3 data (2026-10-05)

v3 = v2 (`data_creation/CLEAN_V2.md`) with **answer-containing random negatives replaced**. Nothing else changes: questions, gold passages, hard negatives, example order, train/dev splits, and the test sets are identical to v2. AmbigQA's v2 data is v1's (`data/training/clean`), so AmbigQA v3 = v1 + this fix.

## Why

v2's random-negative sampler rejects passages whose title equals a gold title or whose unigram overlap with a gold exceeds 30%, but it never checks for answers. 1.23% of QAMPARI and 0.88% of AmbigQA random negatives contained an answer string and acted as false negatives (code/data audit, next-round item N1).

## What `build_clean_v3.py` does

For every example in `data/training/clean_v2{,_hn}/<ds>/{train,dev}_data.jsonl`:

1. Find random negatives (`negative_ctxs`) whose text contains any answer alias, using the same normalization, tokenizer and token-sequence match as `eval.py` / `src.eval_utils.has_answer` (QAMPARI: `answer_list[*].aliases`; AmbigQA: `answers`). Implemented as a faster equivalent: identical verdicts to `has_answer` on 20,000 sampled negatives. One deliberate difference: an alias that tokenizes to nothing (punctuation only) would make `has_answer` match every text; such aliases are skipped (3 QAMPARI train examples).
2. Replace each with a random corpus passage (CSV-parsed `chunks_v5.tsv`) that passes the original filters (title not a gold title; overlap with every gold ≤ 30%) **and** the answer filter, and is not already among the example's negatives. The RNG is seeded from (dataset, split, example id), so the plain and the `_hn` file get identical replacements.

Outputs: `data/training/clean_v3{,_hn}/<ds>/{train,dev}_data.jsonl`, `data/training/clean_v3/<ds>/stats_v3.json`. Dev evaluation files are the v2 ones (`data/phaseA/<ds>_cleanv2dev500.jsonl`): dev questions and golds are unchanged.

Run (repo root, compute node; ~2 min with 8 CPUs after the corpus index):

```bash
python data_creation/build_clean_v3.py
```

## Counts

| Dataset | Split | Examples | Random negatives | Replaced | Examples affected |
| --- | --- | --- | --- | --- | --- |
| QAMPARI | train | 25,869 | 646,725 | 7,946 (1.23%) | 4,026 |
| QAMPARI | dev | 500 | 12,500 | 178 (1.42%) | 93 |
| AmbigQA | train | 3,958 | 98,950 | 872 (0.88%) | 520 |
| AmbigQA | dev | 500 | 12,500 | 115 (0.92%) | 67 |

## Verification (2026-10-05)

An independent check over every file found: 0 answer-containing random negatives left (fast matcher on all; `has_answer` itself on 7,500 sampled negatives per file); every field other than `negative_ctxs` identical to v2; only answer-containing negatives changed; hard negatives unchanged; the `_hn` twin's random negatives identical to the plain file's; 3,000 replacement passages byte-identical to the CSV-parsed corpus; the 14 test files byte-identical (md5) to the baseline recorded before the v1 build.
