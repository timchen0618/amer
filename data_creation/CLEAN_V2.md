# Clean v2 data (QAMPARI rebuild, 2026-10-02)

This document lets a fresh agent rebuild the v2 training and dev data exactly and check the result. A copy sits at `data/training/clean_v2/README.md` (the `data/` tree is not tracked by git; this file is).

## What changed from v1, and why

v1 is `data/training/clean/` (built by `data_creation/build_clean_splits.py`, used by the 24-run grid, git tag `grid-clean-v1`). v2 rebuilds **QAMPARI only**:

| Change | Why |
| --- | --- |
| Train/dev split grouped by shared normalized question **or** gold-set Jaccard ≥ 0.8 (union-find) | v1 grouped by question text only; 127 of 500 v1 dev questions had a train example with the identical gold set (audit G1) |
| Train examples near-duplicating any test example (gold Jaccard ≥ 0.8) removed | 44 of 531 test questions had such a train example (audit G2) |
| Question normalization also strips articles (a, an, the) | article-only differences escaped v1's exact-question checks (audit G5) |
| Gold-count rule on **unique** gold passages, 1–8 | v1 required 5–8 **raw** gold slots, repeats included, which dropped examples with more than 8 slots but at most 8 distinct passages |

Not changed: the test set (`data/amer_data/eval_data/qampari.jsonl`), the corpus, random-negative sampling settings, hard-negative mining settings, and **AmbigQA**. `data/training/clean_v2/ambigqa` and `data/training/clean_v2_hn/ambigqa` are symlinks to the v1 directories, and `data/phaseA/ambigqa_cleanv2dev500.jsonl` is byte-identical to `ambigqa_cleandev500.jsonl` (checked by the verifier). AmbigQA was left alone because the audit found only small residuals there (G5: 8 dev examples share a train gold set; at most 2 test questions match a train question once articles are stripped), and most AmbigQA golds have no passage id, so gold-set grouping does not apply.

Decisions behind the defaults (all recorded in `stats.json` → `args`): the lower bound of 1 unique gold keeps single-gold examples, as v1 did; Jaccard threshold 0.8; dev size 500; split seed 42; seed 12345 for negatives of examples new to v2.

## Inputs

| Input | Role |
| --- | --- |
| `data/training/raw/qampari/train_data_gt_qampari_corpus.jsonl` | the only source of v2 examples (61,911 raw QAMPARI training examples, unique qids) |
| `data/amer_data/eval_data/qampari.jsonl` | test set; read only, for exclusion |
| `data/training/raw/qampari/dev_data_gt_qampari_corpus.jsonl` | QAMPARI dev release the test set is drawn from; read only, for exclusion |
| `data/training/clean/qampari/{train,dev}_data.jsonl`, `data/training/filtered/qampari/{train,dev}_data.jsonl` | sources of reused random negatives, looked up by qid in this order |
| `/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv` | corpus (25,856,230 passages): new negatives, base retrieval, reduced corpus |
| `/scratch/hc3337/embeddings/inf/qampari_embeddings/*` | base-retriever corpus embeddings (32 shards) for the base retrieval |

Checksums of every input and output are in the last section.

## How to rebuild

From the repo root, at the commit that added this file:

```bash
bash data_creation/build_clean_v2.sh
```

It submits four dependent SLURM jobs; the last one's log, `sbatch_outputs/clean_v2_downstream.out`, must end with `ALL CHECKS PASSED` and `ALL_DONE`. The steps, in order:

### 1. Splits: `python data_creation/build_clean_v2.py` (CPU, ~96 GB RAM, ~30 min)

Writes `data/training/clean_v2/qampari/{train_data,dev_data}.jsonl`, `stats.json` (counts after every step) and `provenance.json` (where each example's negatives came from). Then `ln -sfn ../clean/ambigqa data/training/clean_v2/ambigqa`. Inside the script:

1. Clean every gold passage: strip title/text whitespace, undo real CSV quoting. (Raw QAMPARI golds carry a trailing space on the title and a leading space on the text that leaks the label.)
2. Drop examples with an empty question.
3. Keep examples with 1–8 unique gold passage ids.
4. Drop an example if, against any example of the test set or the QAMPARI dev release, it shares the normalized question (lower-case, punctuation removed, articles removed), the qid, or a gold set with Jaccard ≥ 0.8.
5. Drop exact duplicates by (normalized question, unique gold-id set), keeping the first in raw-file order.
6. Union-find over the remaining examples: link two examples if they share a normalized question or their gold sets have Jaccard ≥ 0.8. The build stops if a group exceeds 200 examples.
7. Shuffle groups (seed 42, from a deterministic order), fill dev greedily to exactly 500 examples; the rest is train. Both files keep raw-file order.
8. Train only: drop repeated gold passages within an example. Dev keeps them, as v1 did (evaluation scores each answer's cluster).
9. Random negatives: reuse the example's 25 negatives from the first source file that has its qid (cleaned as in 1); examples new to v2 get 25 sampled with `sample_negatives_and_split.sample_negatives` (rejects passages whose title equals a gold title or whose unigram overlap with a gold exceeds 30%), `random.Random(12345)`, in raw-file order.
10. `hard_negative_ctxs = []`; `positive_ctxs = ground_truths`.

### 2. Dev eval files: `python training/inf_retriever/tools/phaseA_make_dev_sets.py --src_root data/training/clean_v2 --tag cleanv2 --out_dir data/phaseA` (CPU, seconds)

Writes `data/phaseA/qampari_cleanv2dev500.jsonl` (gold passages grouped one per answer, as in the test file) and `data/phaseA/ambigqa_cleanv2dev500.jsonl`.

### 3. Base retrieval: `sbatch data_creation/clean_v2_base_retrieval.sbatch` (1 GPU, ~20 min)

The untrained `infly/inf-retriever-v1-1.5b` over the full corpus, with `retrieval_base.py` (16 shard groups): top-1,000 for the v2 dev set → `results/phaseA/base/qampari_cleanv2dev500.jsonl`; top-100 for the v2 train set → `results/phaseC/mining/qampari_cleanv2train_base_top100.jsonl`.

### 4. Downstream: `sbatch data_creation/clean_v2_downstream.sbatch` (CPU, ~96 GB RAM, ~10 min)

1. Hard negatives: `phaseC_mine_hard_negatives.py --ds qampari --src data/training/clean_v2/qampari --dst data/training/clean_v2_hn/qampari --mined results/phaseC/mining/qampari_cleanv2train_base_top100.jsonl` (defaults: skip the top 3, keep at most 30; reject golds, passages whose title matches a gold title or an answer entity, and passages containing an answer). Dev is copied unchanged. `data/training/clean_v2_hn/ambigqa` links to `../clean_hn/ambigqa`.
2. Reduced corpus for cheap dev evaluation: `phaseA_build_reduced_corpus.py --spec qampari_cleanv2dev500:results/phaseA/base/qampari_cleanv2dev500.jsonl:data/phaseA/qampari_cleanv2dev500.jsonl --topk 1000 --n_random 100000 --seed 0` → `data/phaseA/corpora/qampari_cleanv2dev500.tsv` (base top-1,000 per dev query + all dev golds + the same 100k random passages as every other reduced corpus). AmbigQA's v2 corpus and base run are links to the v1 ones.
3. `python data_creation/verify_clean_v2.py --stage all`.

## Verification (`data_creation/verify_clean_v2.py`)

Re-implements normalization and Jaccard independently of the builder and asserts:

- dev has exactly 500 examples; counts match `stats.json`;
- 0 overlap among train, dev and test (test set + QAMPARI dev release) by normalized question, by qid, and by gold-set Jaccard ≥ 0.8 (train→dev, train→test, dev→test);
- train golds unique, 1–8 per example; dev 1–8 unique golds;
- 25 random negatives per example, none a gold; no passage with stray whitespace;
- reused negatives equal their source file (300 sampled examples);
- the dev eval file has the same 500 questions and golds as the dev split; AmbigQA v2 dev file byte-identical to v1;
- `clean_v2_hn` identical to `clean_v2` except `hard_negative_ctxs`; hard negatives never a gold, never repeated;
- the reduced corpus contains every dev gold id.

## Using v2

```bash
DATA_ROOT=data/training/clean_v2 DEV_TAG=cleanv2 PREFIX=<name> bash training/inf_retriever/tools/grid_launch.sh [filter]
QUEUE=results/phaseC/full_queue_<ds>_<name>.txt DEV_TAG=cleanv2 WORKER=a bash training/inf_retriever/tools/full_eval_queue.sh <ds>
```

`DATA_ROOT` selects the training data (hard-negative runs use `${DATA_ROOT}_hn`), `DEV_TAG` the cheap-eval and full-eval dev sets, and `PREFIX` the run names and the full-eval queue file. Defaults reproduce the 24-run grid on v1.

## Results of the build

From `data/training/clean_v2/qampari/stats.json` and the downstream log (build of 2026-10-02):

| Step | QAMPARI examples |
| --- | --- |
| raw training release | 61,911 |
| after dropping empty questions | 61,901 |
| after the 1–8 unique-gold rule | 33,044 (32,013 also kept by v1's 5–8 raw-slot rule; **1,031 new**, all with more than 8 raw slots; no raw example has fewer than 5 slots) |
| after test exclusion | 32,916 (128 dropped, all by gold-set Jaccard ≥ 0.8; 0 by question or qid) |
| after exact-duplicate removal | 26,369 (6,547 dropped) |
| groups for the split | 21,028 (9,025 examples in groups of 2+; largest group 9 examples; 5,258 gold-Jaccard links, 83 same-question links) |
| **train** | **25,869** (v1: 25,047; 885 new in v2) |
| **dev** | **500** (11 new in v2) |

| | Train | Dev |
| --- | --- | --- |
| unique golds per example (1/2/3/4/5/6/7/8) | 271 / 317 / 348 / 802 / 9,824 / 6,265 / 4,308 / 3,734 | 8 / 6 / 7 / 17 / 187 / 109 / 101 / 65 |
| question type (wikidata_simple / comp / intersection / wikitables_composition) | 13,583 / 7,634 / 1,621 / 3,031 | 255 / 151 / 32 / 62 |

- Train examples that had repeated golds before deduplication: 3,499.
- Random negatives: reused for 25,473 examples (clean train 21,856; clean dev 438; filtered train 2,629; filtered dev 550), freshly sampled for 896.
- Hard negatives (`clean_v2_hn`): 764,567 in total, 29.6 per example on average; 82 examples have none (they always get random negatives). Mining dropped 41,981 gold, 93,377 same-title/answer-entity and 138,714 answer-containing candidates.
- Reduced dev corpus: 461,947 passages (362,476 base top-1,000 candidates, 2,321 dev gold ids of which 930 were not among the candidates, 100,000 random).
- `verify_clean_v2.py --stage all`: all checks passed.

## Checksums (sha256)

Inputs:

```
7ef66fba6a4999392fdf6c8c1b7e01284ec165227421d377c08b041d72d86c96  data/training/raw/qampari/train_data_gt_qampari_corpus.jsonl
6048ba57c4c21063d4947fb14edb471bd1e624d94990c4f025017c58843d93a0  data/amer_data/eval_data/qampari.jsonl
ee54cc7de5e37a8a75e4a629264ddeb8ec863cd3f57a439d4a11c313a4ddd844  data/training/raw/qampari/dev_data_gt_qampari_corpus.jsonl
94a07a56bf101899bbaea163c27717ab8d32a335f0ed435164755dae99fd0010  data/training/clean/qampari/train_data.jsonl
ecb56a051b780abb8caa2745d46588cbc9743aef2845946885924d2a25502af2  data/training/clean/qampari/dev_data.jsonl
ea850872e8bd59c55699e445a7f44dbc312274dbd8ec3cca4373429932f1403c  data/training/filtered/qampari/train_data.jsonl
c18a6b8d7e9a1f6b153f0736c79234debe859aa3d146753827d8fc510b6fb3c8  data/training/filtered/qampari/dev_data.jsonl
fb1844364c0f55b8b409b22d82656b3e39cfb7f7b683c697f2828fa2e117c129  /scratch/hc3337/wikipedia_chunks/chunks_v5.tsv
```

Outputs. Steps 1–2 are deterministic and should reproduce these hashes exactly. Step 3 runs FAISS on a GPU, whose float results can differ slightly by GPU type; the base-retrieval files, the hard negatives and the reduced corpus derived from them may then differ in hash, which is expected. Compare their counts above and the verifier's result instead.

```
085fe5b22af131ae33890fd561a6ffd060aceb74e96d9e7f0b45bbca998852d7  data/training/clean_v2/qampari/train_data.jsonl
1fa8bb4df886c4825caac9a39af4295fa6ebeae3b2612c5fa1526ec62c6b7de0  data/training/clean_v2/qampari/dev_data.jsonl
0a93e6beab4c924eb32b3dd79716d63cb4099ff08969e5d23788441aa47ddbaf  data/training/clean_v2/qampari/stats.json
300f59504e5a1b183a9484663731acac1b16d1d7bfcaedbc0e568eb0c8e0ef1b  data/training/clean_v2/qampari/provenance.json
13884dbb3a2346fac4a04cbbf387310abdf3ba693618905f1b45162cc63ed706  data/phaseA/qampari_cleanv2dev500.jsonl
1032abf6feac63bdedafd01ca68568af15111555f4b9b7e5df0ad154170dd468  data/phaseA/ambigqa_cleanv2dev500.jsonl
bd2033043b3ad214722ebe7dfa149fbc7c7a23c60efdd59d1fb79b9d04a0706e  results/phaseA/base/qampari_cleanv2dev500.jsonl
244dfdb543019680ce0dbdb7444e26db83d1b90e5855358d354eb57502e7b599  results/phaseC/mining/qampari_cleanv2train_base_top100.jsonl
5d1c66bee8f67ab25fffc5c8a4aa1069117fd64b5ce3441ec6b4a2b5a4096f46  data/training/clean_v2_hn/qampari/train_data.jsonl
1fa8bb4df886c4825caac9a39af4295fa6ebeae3b2612c5fa1526ec62c6b7de0  data/training/clean_v2_hn/qampari/dev_data.jsonl
da70158a3d4c36af0f6360a6944f2ac0527f46b88ef1b285aa14c792136d7aae  data/phaseA/corpora/qampari_cleanv2dev500.tsv
```
