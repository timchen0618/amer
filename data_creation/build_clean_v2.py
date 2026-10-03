"""Build the v2 QAMPARI train/dev splits (code audit 2026-09-28: G1, G2, G5; gold-count rule
changed 2026-10-02). AmbigQA is not rebuilt: data/training/clean_v2/ambigqa links to
data/training/clean/ambigqa (see data/training/clean_v2/README.md).

Input: the raw QAMPARI training release, data/training/raw/qampari/train_data_gt_qampari_corpus.jsonl
(61,911 examples, unique qids). Test files are only read, never written.

Steps, in order (every count goes to stats.json):
  1. clean every gold passage: strip title/text whitespace and undo CSV quoting (the raw QAMPARI
     golds carry a trailing/leading space that leaks the label; see step5_progress_2026-09-27.md);
  2. drop examples with an empty question;
  3. gold-count rule: keep examples with --min_golds..--max_golds UNIQUE gold passages (by id).
     The v1 build kept 5-8 RAW gold slots, repeats included, which excluded examples with more
     than 8 slots but at most 8 distinct passages;
  4. test exclusion: drop an example if, against any example of the test set or of the QAMPARI dev
     release the test set is drawn from, it shares (a) the normalized question (lower-case, no
     punctuation, no articles), (b) the qid, or (c) a gold set with Jaccard >= --jaccard;
  5. drop exact duplicates, keyed by (normalized question, set of unique gold ids); the first copy
     in raw-file order is kept;
  6. group examples with union-find: two examples are linked if they share a normalized question
     or their gold sets have Jaccard >= --jaccard (paraphrases and sibling templates over the same
     passages, audit G1). The build stops if the largest group exceeds --max_group;
  7. split by group (seeded shuffle of groups, greedy fill) into exactly --dev_size dev examples
     and the rest as train, so no group spans both sides;
  8. train only: drop repeated gold passages within an example (dev keeps them, as in v1; eval
     scores each answer's cluster);
  9. random negatives: reuse the example's existing 25 negatives, looked up by qid in
     data/training/clean/qampari/{train,dev}_data.jsonl, else in data/training/filtered/qampari/
     {train,dev}_data.jsonl (cleaned as in step 1). Examples new to v2 get 25 freshly sampled ones,
     with sample_negatives_and_split.sample_negatives (same title / 30% unigram-overlap filters as
     v1), random.Random(--neg_seed), in raw-file order;
 10. clean every negative passage as in step 1; set hard_negative_ctxs = [].
Output: <out_dir>/{train_data,dev_data}.jsonl, stats.json and provenance.json (per-example source
of the negatives).

Usage (repo root, compute node, ~64 GB RAM; step 9's corpus index reads the 15 GB corpus once):
    python data_creation/build_clean_v2.py
"""
import argparse
import collections
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sample_negatives_and_split import build_offset_index, sample_negatives  # noqa: E402

RAW = "data/training/raw/qampari/train_data_gt_qampari_corpus.jsonl"
TEST_FILES = ["data/amer_data/eval_data/qampari.jsonl",
              "data/training/raw/qampari/dev_data_gt_qampari_corpus.jsonl"]
NEG_SOURCES = ["data/training/clean/qampari/train_data.jsonl", "data/training/clean/qampari/dev_data.jsonl",
               "data/training/filtered/qampari/train_data.jsonl", "data/training/filtered/qampari/dev_data.jsonl"]
CORPUS = "/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv"
ARTICLES = {"a", "an", "the"}


def norm_q(q):
    words = re.sub(r"[^\w\s]", "", (q or "").lower()).split()
    return " ".join(w for w in words if w not in ARTICLES)


def csv_unquote(s):
    # only a real CSV-quoted field (doubled inner quotes, no single one); see finetuning_data._csv_unquote
    inner = s[1:-1]
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"' and '""' in inner and '"' not in inner.replace('""', ''):
        return s[1:-1].replace('""', '"')
    return s


def clean_passage(p):
    p = dict(p)
    p["title"] = csv_unquote((p.get("title") or "").strip()).strip()
    p["text"] = csv_unquote((p.get("text") or "").strip()).strip()
    return p


def flat_golds(ex):
    g = ex["ground_truths"]
    return [p for grp in g for p in grp] if g and isinstance(g[0], list) else g


def gold_set(ex):
    return frozenset(str(p["id"]) for p in flat_golds(ex))


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 0.0


class InvIndex:
    """gold id -> example indices, for near-duplicate gold-set lookups."""
    def __init__(self, sets):
        self.sets, self.inv = sets, collections.defaultdict(list)
        for i, s in enumerate(sets):
            for g in s:
                self.inv[g].append(i)

    def near(self, s, thr):
        cand = {i for g in s for i in self.inv.get(g, ())}
        return [i for i in cand if jaccard(s, self.sets[i]) >= thr]


def hist(values):
    return dict(sorted(collections.Counter(values).items()))


def find(parent, i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/training/clean_v2/qampari")
    ap.add_argument("--min_golds", type=int, default=1)
    ap.add_argument("--max_golds", type=int, default=8)
    ap.add_argument("--jaccard", type=float, default=0.8)
    ap.add_argument("--dev_size", type=int, default=500)
    ap.add_argument("--max_group", type=int, default=200)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--neg_seed", type=int, default=12345)
    ap.add_argument("--num_negatives", type=int, default=25)
    ap.add_argument("--overlap_threshold", type=float, default=0.3)
    args = ap.parse_args()
    st = collections.OrderedDict()
    st["args"] = vars(args)

    data = [json.loads(l) for l in open(RAW)]
    st["raw_examples"] = len(data)
    assert len({ex["qid"] for ex in data}) == len(data), "raw qids are not unique"

    # 1. clean golds
    for ex in data:
        ex["ground_truths"] = [clean_passage(p) for p in ex["ground_truths"]]

    # 2. empty questions
    data = [ex for ex in data if norm_q(ex["question_text"])]
    st["after_drop_empty_question"] = len(data)

    # 3. gold-count rule on unique golds
    slots = [len(ex["ground_truths"]) for ex in data]
    keep = [args.min_golds <= len(gold_set(ex)) <= args.max_golds for ex in data]
    st["new_vs_v1_rule"] = {
        "kept_by_both (5-8 raw slots)": sum(k and 5 <= s <= 8 for k, s in zip(keep, slots)),
        "new_in_v2 (>8 raw slots, <=8 unique)": sum(k and not 5 <= s <= 8 for k, s in zip(keep, slots)),
        "kept_by_v1_only": sum((not k) and 5 <= s <= 8 for k, s in zip(keep, slots)),
    }
    data = [ex for ex, k in zip(data, keep) if k]
    for ex in data:
        ex["_new_in_v2"] = not 5 <= len(ex["ground_truths"]) <= 8
    st["after_gold_count_rule"] = len(data)

    # 4. test exclusion
    test = [json.loads(l) for f in TEST_FILES for l in open(f)]
    test_q = {norm_q(t["question_text"]) for t in test}
    test_ids = {t["qid"] for t in test}
    test_inv = InvIndex([gold_set(t) for t in test])
    reasons = collections.Counter()
    kept = []
    for ex in data:
        r = ("question" if norm_q(ex["question_text"]) in test_q else
             "qid" if ex["qid"] in test_ids else
             "gold_jaccard" if test_inv.near(gold_set(ex), args.jaccard) else None)
        if r:
            reasons[r] += 1
        else:
            kept.append(ex)
    st["dropped_test_overlap"] = dict(reasons)
    data = kept
    st["after_test_exclusion"] = len(data)

    # 5. exact duplicates
    seen, dedup = set(), []
    for ex in data:
        key = (norm_q(ex["question_text"]), gold_set(ex))
        if key not in seen:
            seen.add(key)
            dedup.append(ex)
    st["dropped_exact_duplicates"] = len(data) - len(dedup)
    data = dedup
    st["after_dedup"] = len(data)

    # 6. union-find groups
    n = len(data)
    parent = list(range(n))
    sets = [gold_set(ex) for ex in data]
    inv = InvIndex(sets)
    by_q = collections.defaultdict(list)
    for i, ex in enumerate(data):
        by_q[norm_q(ex["question_text"])].append(i)
    links = collections.Counter()
    for idx in by_q.values():
        for j in idx[1:]:
            a, b = find(parent, idx[0]), find(parent, j)
            if a != b:
                parent[b] = a
                links["question"] += 1
    for i in range(n):
        for j in inv.near(sets[i], args.jaccard):
            if j > i:
                a, b = find(parent, i), find(parent, j)
                if a != b:
                    parent[b] = a
                    links["gold_jaccard"] += 1
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(parent, i)].append(i)
    sizes = [len(g) for g in groups.values()]
    st["groups"] = {"count": len(groups), "examples_in_multi_groups": sum(s for s in sizes if s > 1),
                    "largest": max(sizes), "size_hist": hist(sizes), "merges": dict(links)}
    assert max(sizes) <= args.max_group, f"largest group has {max(sizes)} examples (> --max_group)"

    # 7. split by group
    keys = sorted(groups, key=lambda r: min(groups[r]))  # deterministic order before the shuffle
    random.Random(args.split_seed).shuffle(keys)
    dev_idx, train_idx = [], []
    for k in keys:
        g = groups[k]
        (dev_idx if len(dev_idx) + len(g) <= args.dev_size else train_idx).extend(g)
    assert len(dev_idx) == args.dev_size, f"could not fill dev exactly: {len(dev_idx)}"
    dev_idx.sort(); train_idx.sort()  # raw-file order within each split
    dev, train = [data[i] for i in dev_idx], [data[i] for i in train_idx]

    # 8. train-only gold dedup
    st["train_examples_with_repeated_golds"] = sum(len(ex["ground_truths"]) != len(gold_set(ex)) for ex in train)
    st["train_gold_slots_hist_before_dedup"] = hist(len(ex["ground_truths"]) for ex in train)
    for ex in train:
        uniq, seen_g = [], set()
        for p in ex["ground_truths"]:
            if str(p["id"]) not in seen_g:
                seen_g.add(str(p["id"]))
                uniq.append(p)
        ex["ground_truths"] = uniq
    for ex in train + dev:
        ex["positive_ctxs"] = list(ex["ground_truths"])

    # 9-10. negatives
    pool = {}
    for f in NEG_SOURCES:
        for l in open(f):
            e = json.loads(l)
            pool.setdefault(e["qid"], (f, e["negative_ctxs"]))
    prov = {}
    raw_order = {ex["qid"]: i for i, ex in enumerate(data)}
    need = sorted((ex for ex in train + dev if ex["qid"] not in pool), key=lambda e: raw_order[e["qid"]])
    if need:
        offsets = build_offset_index(CORPUS)
        rng = random.Random(args.neg_seed)
        with open(CORPUS, "rb") as fh:
            for ex in need:
                negs = sample_negatives(fh, offsets, ex["ground_truths"], args.num_negatives,
                                        args.overlap_threshold, rng)
                assert len(negs) == args.num_negatives, f"{ex['qid']}: only {len(negs)} negatives"
                pool[ex["qid"]] = ("sampled", negs)
    for ex in train + dev:
        src, negs = pool[ex["qid"]]
        ex["negative_ctxs"] = [clean_passage(p) for p in negs]
        ex["hard_negative_ctxs"] = []
        prov[ex["qid"]] = src
    st["negatives_source"] = dict(collections.Counter(prov.values()))

    # sanity checks
    tr_q = {norm_q(e["question_text"]) for e in train}
    dv_q = {norm_q(e["question_text"]) for e in dev}
    assert not tr_q & dv_q, "train/dev question overlap"
    assert not (tr_q | dv_q) & test_q, "test question overlap"
    assert not {e["qid"] for e in train + dev} & test_ids, "test qid overlap"
    dev_inv = InvIndex([gold_set(e) for e in dev])
    assert not any(dev_inv.near(gold_set(e), args.jaccard) for e in train), "train/dev gold-set near-duplicate"
    for e in train:
        assert len({p["id"] for p in e["ground_truths"]}) == len(e["ground_truths"])
        assert args.min_golds <= len(e["ground_truths"]) <= args.max_golds
    for e in train + dev:
        for p in e["ground_truths"] + e["negative_ctxs"]:
            assert p["title"] == p["title"].strip() and p["text"] == p["text"].strip()

    st["train_examples"] = len(train)
    st["dev_examples"] = len(dev)
    st["train_new_in_v2"] = sum(e["_new_in_v2"] for e in train)
    st["dev_new_in_v2"] = sum(e["_new_in_v2"] for e in dev)
    st["train_unique_golds_hist"] = hist(len(e["ground_truths"]) for e in train)
    st["dev_unique_golds_hist"] = hist(len(gold_set(e)) for e in dev)
    st["dev_gold_slots_hist"] = hist(len(e["ground_truths"]) for e in dev)
    st["train_question_type"] = hist(e["qid"].split("__")[1] for e in train)
    st["dev_question_type"] = hist(e["qid"].split("__")[1] for e in dev)

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (("train_data", train), ("dev_data", dev)):
        with open(f"{args.out_dir}/{name}.jsonl", "w") as f:
            for ex in rows:
                ex.pop("_new_in_v2", None)
                f.write(json.dumps(ex) + "\n")
    json.dump(st, open(f"{args.out_dir}/stats.json", "w"), indent=2)
    json.dump(prov, open(f"{args.out_dir}/provenance.json", "w"))
    print(json.dumps(st, indent=2))


if __name__ == "__main__":
    main()
