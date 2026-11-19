"""Build clean train/dev splits for the INF-Retriever multi-query pipeline (2026-09-27 audit).

Input: the union of data/training/filtered/<ds>/{train,dev}_data.jsonl, i.e. every raw training
example that passed the gold-count filter and got its random negatives from
sample_negatives_and_split.py. Steps, in order (counts go to stats.json):
  1. drop examples with an empty question;
  2. drop examples whose normalized question or id appears in any test file (QAMPARI: the test
     set and the whole QAMPARI dev release it is drawn from; AmbigQA: the test set). Test files
     are only read, never written;
  3. drop exact duplicates, keyed by (normalized question, set of gold passages);
  4. split by normalized question, so every copy of a question lands on the same side, into a dev
     set of exactly --dev_size examples and the rest as train;
  5. train only: drop repeated gold passages within an example (same id, or same title + text when
     there is no id); AmbigQA training examples left with fewer than 2 golds are dropped;
  6. every passage (golds, negatives): strip title/text whitespace and undo CSV quoting, so the
     files match the csv-parsed corpus string that gen_embed_new.py embeds.
Output: data/training/clean/<ds>/{train_data,dev_data}.jsonl and stats.json.

Usage (repo root, compute node):
    python data_creation/build_clean_splits.py --ds qampari
"""
import argparse
import collections
import json
import os
import random
import re

TEST_FILES = {
    "qampari": ["data/amer_data/eval_data/qampari.jsonl",
                "data/training/raw/qampari/dev_data_gt_qampari_corpus.jsonl"],
    "ambigqa": ["data/amer_data/eval_data/ambigqa.jsonl"],
}


def norm_q(q):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (q or "").lower())).strip()


def question(ex):
    return ex.get("question_text") or ex.get("question") or ""


def ex_id(ex):
    return str(ex.get("qid") or ex.get("id") or "")


def csv_unquote(s):
    # only a real CSV-quoted field (doubled inner quotes, no single one); see finetuning_data._csv_unquote
    inner = s[1:-1]
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"' and '""' in inner and '"' not in inner.replace('""', ''):
        return s[1:-1].replace('""', '"')
    return s


def clean_passage(p):
    p = dict(p)
    p["title"] = csv_unquote(p.get("title", "").strip()).strip()
    p["text"] = csv_unquote(p.get("text", "").strip()).strip()
    return p


def gold_key(p):
    return p.get("id") or (p.get("title", "").strip() + "\t" + p.get("text", "").strip())


def hist(values):
    return dict(sorted(collections.Counter(values).items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["qampari", "ambigqa"])
    ap.add_argument("--src_dir", default=None, help="default: data/training/filtered/<ds>")
    ap.add_argument("--out_dir", default=None, help="default: data/training/clean/<ds>")
    ap.add_argument("--dev_size", type=int, default=500)
    ap.add_argument("--min_train_golds", type=int, default=None,
                    help="drop training examples with fewer unique golds (default: 2 for ambigqa, off for qampari)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    src = args.src_dir or f"data/training/filtered/{args.ds}"
    out = args.out_dir or f"data/training/clean/{args.ds}"
    min_golds = args.min_train_golds if args.min_train_golds is not None else (2 if args.ds == "ambigqa" else 0)
    stats = collections.OrderedDict()

    data = []
    for split in ("train", "dev"):
        data += [json.loads(l) for l in open(f"{src}/{split}_data.jsonl")]
    stats["input_examples"] = len(data)

    # 1. empty questions
    data = [ex for ex in data if norm_q(question(ex))]
    stats["after_drop_empty_question"] = len(data)

    # 2. test overlap
    test_q, test_ids = set(), set()
    for f in TEST_FILES[args.ds]:
        for l in open(f):
            t = json.loads(l)
            test_q.add(norm_q(question(t)))
            if ex_id(t):
                test_ids.add(ex_id(t))
    before = len(data)
    data = [ex for ex in data if norm_q(question(ex)) not in test_q and ex_id(ex) not in test_ids]
    stats["dropped_test_overlap"] = before - len(data)
    stats["after_drop_test_overlap"] = len(data)

    # 3. exact duplicates: same question and same set of gold passages
    seen, dedup = set(), []
    for ex in data:
        key = (norm_q(question(ex)), frozenset(gold_key(p) for p in ex["positive_ctxs"]))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(ex)
    stats["dropped_exact_duplicates"] = len(data) - len(dedup)
    data = dedup
    stats["after_dedup"] = len(data)

    # 4. split by question
    groups = collections.OrderedDict()
    for ex in data:
        groups.setdefault(norm_q(question(ex)), []).append(ex)
    keys = list(groups)
    random.Random(args.seed).shuffle(keys)
    dev, train = [], []
    for k in keys:
        g = groups[k]
        if len(dev) + len(g) <= args.dev_size:
            dev += g
        else:
            train += g
    assert len(dev) == args.dev_size, f"could not fill dev exactly: {len(dev)}"
    stats["questions_sharing_text_groups"] = sum(len(g) > 1 for g in groups.values())

    # 5. train only: drop repeated golds within an example
    before_counts = [len(ex["positive_ctxs"]) for ex in train]
    changed, kept = 0, []
    for ex in train:
        uniq, seen_g = [], set()
        for p in ex["positive_ctxs"]:
            k = gold_key(p)
            if k in seen_g:
                continue
            seen_g.add(k)
            uniq.append(p)
        changed += len(uniq) < len(ex["positive_ctxs"])
        ex["positive_ctxs"] = uniq
        if "ground_truths" in ex:
            ex["ground_truths"] = uniq
        if len(uniq) >= min_golds:
            kept.append(ex)
    after_counts_all = [len(ex["positive_ctxs"]) for ex in train]
    stats["train_examples_with_repeated_golds"] = changed
    stats["train_golds_per_example_before"] = {"mean": round(sum(before_counts) / len(before_counts), 3),
                                               "hist": hist(before_counts)}
    stats["train_golds_per_example_after_dedup"] = {"mean": round(sum(after_counts_all) / len(after_counts_all), 3),
                                                    "hist": hist(after_counts_all)}
    stats["train_dropped_below_min_golds"] = len(train) - len(kept)
    stats["min_train_golds"] = min_golds
    train = kept
    kept_counts = [len(ex["positive_ctxs"]) for ex in train]
    stats["train_golds_per_example_final"] = {"mean": round(sum(kept_counts) / len(kept_counts), 3),
                                              "hist": hist(kept_counts)}

    # 6. clean every passage
    for ex in train + dev:
        for k in ("positive_ctxs", "ground_truths", "negative_ctxs", "hard_negative_ctxs"):
            if isinstance(ex.get(k), list):
                ex[k] = [clean_passage(p) if isinstance(p, dict) else p for p in ex[k]]

    # sanity: no question crosses train/dev or reaches the test set
    tr_q = {norm_q(question(e)) for e in train}
    dv_q = {norm_q(question(e)) for e in dev}
    assert not (tr_q & dv_q), "train/dev question overlap"
    assert not ((tr_q | dv_q) & test_q), "test question overlap"
    assert not ({ex_id(e) for e in train + dev} & test_ids), "test id overlap"
    stats["train_examples"] = len(train)
    stats["dev_examples"] = len(dev)
    stats["dev_golds_per_example"] = hist(len(ex["positive_ctxs"]) for ex in dev)

    os.makedirs(out, exist_ok=True)
    for name, rows in (("train_data", train), ("dev_data", dev)):
        with open(f"{out}/{name}.jsonl", "w") as f:
            for ex in rows:
                f.write(json.dumps(ex) + "\n")
    json.dump(stats, open(f"{out}/stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
