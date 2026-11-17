# Slimmed retrieval outputs (2026-11-13)

The retrieval outputs `<ds>/<run>_s<step>/{dev,test}/<ds>.jsonl` of finished runs have been slimmed in place. Each line
now keeps only the query identity (`qid`/`id`, `question_text`/`question`), `"_slim": 1`, and `ctxs` with `id` and
`score`. Passage titles and texts were dropped, which shrank the files to about 8% of their size (about 410 GB in total).
Every file was checked before it was replaced:
- its titles and texts equal the corpus rows of their ids, so the original can be rebuilt exactly;
- the slim copy has the same rows, ids and scores in the same order.

- QAMPARI scoring (`eval.py`, gold ids) and `tools/fusion_combo.py` read slim files directly.
- AmbigQA scoring (`eval.py --no-gold-id`) matches answer strings in passage text, so restore the file first:

      python training/inf_retriever/tools/slim_retrieval.py restore <slim.jsonl> <out.jsonl> --data <eval data jsonl>

  Run this on a compute node; it reads the 15 GB corpus `/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv`. The
  `--data` file is the `Data path:` line in the folder's `eval_metrics.txt`. With `--data`, the output is
  byte-identical to the original.

Test: `training/inf_retriever/tests/test_slim_retrieval.py` (QAMPARI and AmbigQA, dev and test: recorded metrics
reproduced on the slim file for QAMPARI and on the restored file for both; the restored file is byte-identical).
Bulk run: `training/inf_retriever/tools/slim_all.py`. Files of runs that are still training or being evaluated were
skipped.
