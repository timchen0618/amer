"""Seeded 90/10 train/dev split of the clean QAMPARI training pool (objective audit, 2026-10-23, user decision).

The 500-question dev set was too small to select checkpoints reliably. As in the legacy reproduction code
(src/dataset.py: train_test_split(test_size=0.1, seed=42)), this pools data/training/clean_v3_hn/qampari/
{train,dev}_data.jsonl (25869 + 500 questions, same cleaning, hard negatives kept) and holds out 10% with a seed:
  data/training/clean_v3_hn_s<seed>/qampari/train_data.jsonl   training (full records)
  data/training/clean_v3_hn_s<seed>/qampari/dev_data.jsonl     in-training eval set (full records)
  data/phaseA/qampari_clean_v3_dev_s<seed>.jsonl               full-dev retrieval set (the fields the eval reads)
A different seed gives a different split, for variance runs. The test set is not touched.
Only questions with --min_golds..--max_golds UNIQUE gold passages (default 5-8, user 2026-10-27) are kept, before
splitting. Train questions get repeated golds removed (as clean_v2 step 8; old dev records keep their repeats, which
the pool would otherwise carry into train); dev questions keep them, as in the clean dev sets.

    python training/inf_retriever/tools/build_seed_split.py --seed 0
"""
import argparse, json, os, random

SRC = "data/training/clean_v3_hn/qampari"
FULLDEV_KEYS = ("qid", "question_text", "entities", "answer_list", "ground_truths")
OLD_FULLDEV = "data/phaseA/qampari_cleanv2dev500.jsonl"


def fulldev_record(x):
    """The full-dev format: FULLDEV_KEYS, with each gold as a one-passage cluster ([[g1], [g2], ...]), which is
    what the eval scorer reads (a cluster counts as found when any of its ids is retrieved)."""
    r = {k: x[k] for k in FULLDEV_KEYS}
    r["ground_truths"] = [g if isinstance(g, list) else [g] for g in x["ground_truths"]]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev_frac", type=float, default=0.1)
    ap.add_argument("--min_golds", type=int, default=5)
    ap.add_argument("--max_golds", type=int, default=8)
    a = ap.parse_args()
    pool = [json.loads(l) for f in ("train_data.jsonl", "dev_data.jsonl") for l in open(f"{SRC}/{f}")]
    qids = [x["qid"] for x in pool]
    assert len(set(qids)) == len(qids), "duplicate qid in pool"
    # The old full-dev file must be exactly the old dev records restricted to FULLDEV_KEYS (format check).
    old = {x["qid"]: x for x in (json.loads(l) for l in open(OLD_FULLDEV))}
    by_qid = {x["qid"]: x for x in pool}
    for q, rec in old.items():
        assert fulldev_record(by_qid[q]) == rec, f"full-dev format mismatch for qid {q}"
    n_uniq = lambda x: len({str(g["id"]) for g in x["ground_truths"]})
    n_pool = len(pool)
    n_below = sum(n_uniq(x) < a.min_golds for x in pool)
    n_above = sum(n_uniq(x) > a.max_golds for x in pool)
    pool = [x for x in pool if a.min_golds <= n_uniq(x) <= a.max_golds]
    print(f"gold-count filter {a.min_golds}-{a.max_golds} unique golds: pool {n_pool} -> {len(pool)} "
          f"(dropped {n_below} with < {a.min_golds}, {n_above} with > {a.max_golds})")
    rng = random.Random(a.seed)
    order = list(range(len(pool)))
    rng.shuffle(order)
    n_dev = round(a.dev_frac * len(pool))
    dev_idx, train_idx = sorted(order[:n_dev]), sorted(order[n_dev:])
    out = f"data/training/clean_v3_hn_s{a.seed}/qampari"
    os.makedirs(out, exist_ok=True)
    with open(f"{out}/train_data.jsonl", "w") as f:
        for i in train_idx:
            x = dict(pool[i])
            seen, uniq = set(), []
            for g in x["ground_truths"]:
                if str(g["id"]) not in seen:
                    seen.add(str(g["id"])); uniq.append(g)
            x["ground_truths"] = uniq
            x["positive_ctxs"] = list(uniq)
            f.write(json.dumps(x) + "\n")
    # Dev records drop their mined hard negatives, as in the clean dev sets (hard_negative_ctxs = []): the in-training
    # eval then scores against random negatives only, and its single-query branch needs exactly one negative per question.
    with open(f"{out}/dev_data.jsonl", "w") as f:
        for i in dev_idx:
            f.write(json.dumps(dict(pool[i], hard_negative_ctxs=[])) + "\n")
    fd = f"data/phaseA/qampari_clean_v3_dev_s{a.seed}.jsonl"
    with open(fd, "w") as f:
        for i in dev_idx:
            f.write(json.dumps(fulldev_record(pool[i])) + "\n")
    test = {json.loads(l).get("question_text") for l in open("data/amer_data/eval_data/qampari.jsonl")}
    leak = sum(pool[i]["question_text"] in test for i in range(len(pool)))
    print(f"seed {a.seed}: pool {len(pool)} -> train {len(train_idx)}, dev {len(dev_idx)}; "
          f"old full-dev format check OK; pool questions also in test: {leak}")
    print(f"wrote {out}/{{train,dev}}_data.jsonl and {fd}")


if __name__ == "__main__":
    main()
