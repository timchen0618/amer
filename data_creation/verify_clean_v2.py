"""Independent checks of the v2 QAMPARI data (data_creation/build_clean_v2.py and the steps after
it, see data/training/clean_v2/README.md). Every check fails loudly (AssertionError).

    python data_creation/verify_clean_v2.py --stage splits   # after build_clean_v2.py
    python data_creation/verify_clean_v2.py --stage all      # after hard negatives + reduced corpus

Normalization and Jaccard are re-implemented here rather than imported, so a bug in the builder's
version cannot hide itself.
"""
import argparse
import collections
import csv
import hashlib
import json
import random
import re
import sys

csv.field_size_limit(sys.maxsize)
V2 = "data/training/clean_v2/qampari"
HN = "data/training/clean_v2_hn/qampari"
TEST_FILES = ["data/amer_data/eval_data/qampari.jsonl",
              "data/training/raw/qampari/dev_data_gt_qampari_corpus.jsonl"]
DEV_EVAL = "data/phaseA/qampari_cleanv2dev500.jsonl"
CORPUS = "data/phaseA/corpora/qampari_cleanv2dev500.tsv"


def nq(q):
    return " ".join(w for w in re.sub(r"[^\w\s]", "", (q or "").lower()).split() if w not in ("a", "an", "the"))


def golds(ex):
    g = ex["ground_truths"]
    if g and isinstance(g[0], list):
        g = [p for grp in g for p in grp]
    return g


def gset(ex):
    return frozenset(str(p["id"]) for p in golds(ex))


def near_pairs(A, B, thr=0.8):
    """Number of examples in A with a gold set of Jaccard >= thr to some example in B."""
    inv = collections.defaultdict(set)
    for j, s in enumerate(B):
        for g in s:
            inv[g].add(j)
    n = 0
    for s in A:
        cand = set().union(*(inv[g] for g in s)) if s else set()
        n += any(len(s & B[j]) / len(s | B[j]) >= thr for j in cand)
    return n


def load(p):
    return [json.loads(l) for l in open(p)]


def ok(msg):
    print("OK  ", msg, flush=True)


def check_splits():
    tr, dv = load(f"{V2}/train_data.jsonl"), load(f"{V2}/dev_data.jsonl")
    te = [t for f in TEST_FILES for t in load(f)]
    st = json.load(open(f"{V2}/stats.json"))
    assert len(dv) == 500, len(dv); ok("dev has 500 examples")
    assert len(tr) == st["train_examples"] and len(dv) == st["dev_examples"]; ok(f"train {len(tr)} matches stats.json")
    trq, dvq, teq = ({nq(e["question_text"]) for e in x} for x in (tr, dv, te))
    assert not trq & dvq and not trq & teq and not dvq & teq; ok("0 normalized-question overlap among train / dev / test")
    tri, dvi, tei = ({e["qid"] for e in x} for x in (tr, dv, te))
    assert len(tri) == len(tr) and len(dvi) == len(dv); ok("qids unique within train and within dev")
    assert not tri & dvi and not tri & tei and not dvi & tei; ok("0 qid overlap among train / dev / test")
    TR, DV, TE = ([gset(e) for e in x] for x in (tr, dv, te))
    for name, a, b in (("train->dev", TR, DV), ("train->test", TR, TE), ("dev->test", DV, TE)):
        n = near_pairs(a, b)
        assert n == 0, f"{name}: {n} gold-set near-duplicates"
    ok("0 gold-set near-duplicates (Jaccard >= 0.8) train->dev, train->test, dev->test")
    for e in tr:
        ids = [str(p["id"]) for p in golds(e)]
        assert len(ids) == len(set(ids)) and 1 <= len(ids) <= 8, e["qid"]
        assert e["positive_ctxs"] == e["ground_truths"], e["qid"]
    ok("train golds unique, 1-8 per example, positive_ctxs == ground_truths")
    for e in dv:
        assert 1 <= len(gset(e)) <= 8 and e["positive_ctxs"] == e["ground_truths"], e["qid"]
    ok("dev: 1-8 unique golds per example, positive_ctxs == ground_truths")
    ws = 0
    for e in tr + dv:
        assert len(e["negative_ctxs"]) == 25 and e["hard_negative_ctxs"] == [], e["qid"]
        assert not {str(p["id"]) for p in e["negative_ctxs"]} & gset(e), e["qid"]
        for p in golds(e) + e["negative_ctxs"]:
            ws += p["title"] != p["title"].strip() or p["text"] != p["text"].strip()
    assert ws == 0, f"{ws} passages with stray whitespace"
    ok("25 random negatives each, none is a gold, no stray whitespace in any passage")
    # reused negatives equal their source (after the same cleaning), on a seeded sample
    prov = json.load(open(f"{V2}/provenance.json"))
    src_rows = {}
    for f in set(prov.values()) - {"sampled"}:
        for e in load(f):
            src_rows.setdefault(e["qid"], {})[f] = e["negative_ctxs"]
    rng = random.Random(0)
    sample = rng.sample([e for e in tr + dv if prov[e["qid"]] != "sampled"], 300)
    for e in sample:
        src = src_rows[e["qid"]][prov[e["qid"]]]
        assert [p["id"] for p in src] == [p["id"] for p in e["negative_ctxs"]], e["qid"]
    ok(f"reused negatives match their source file (300 sampled examples); sources: {collections.Counter(prov.values())}")
    # dev eval file
    de = load(DEV_EVAL)
    assert [e["qid"] for e in de] and {e["qid"] for e in de} == dvi and len(de) == 500
    by_q = {e["qid"]: e for e in dv}
    for e in de:
        assert [g[0]["id"] for g in e["ground_truths"]] == [p["id"] for p in by_q[e["qid"]]["ground_truths"]]
    ok(f"{DEV_EVAL}: same 500 questions and golds as the v2 dev split")
    a1, a2 = open("data/phaseA/ambigqa_cleandev500.jsonl", "rb").read(), open("data/phaseA/ambigqa_cleanv2dev500.jsonl", "rb").read()
    assert hashlib.sha256(a1).digest() == hashlib.sha256(a2).digest(); ok("AmbigQA v2 dev file is byte-identical to v1")


def check_downstream():
    tr, hn = load(f"{V2}/train_data.jsonl"), load(f"{HN}/train_data.jsonl")
    assert len(tr) == len(hn)
    n_hard, empty = 0, 0
    for a, b in zip(tr, hn):
        assert {k: v for k, v in a.items() if k != "hard_negative_ctxs"} == {k: v for k, v in b.items() if k != "hard_negative_ctxs"}, a["qid"]
        ids = [str(p["id"]) for p in b["hard_negative_ctxs"]]
        assert len(ids) == len(set(ids)) and not set(ids) & gset(a), a["qid"]
        n_hard += len(ids); empty += not ids
    ok(f"clean_v2_hn train == clean_v2 train except hard negatives; {n_hard} hard negatives, "
       f"none a gold or repeated; {empty} examples without any")
    assert open(f"{HN}/dev_data.jsonl", "rb").read() == open(f"{V2}/dev_data.jsonl", "rb").read(); ok("hn dev file identical to v2 dev")
    need = {str(p["id"]) for e in load(DEV_EVAL) for grp in e["ground_truths"] for p in grp}
    have = set()
    with open(CORPUS, newline="") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            have.add(row[0])
    assert need <= have, f"{len(need - have)} dev gold ids missing from the reduced corpus"
    ok(f"reduced corpus ({len(have)} passages) contains all {len(need)} dev gold ids")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["splits", "all"], default="all")
    a = ap.parse_args()
    check_splits()
    if a.stage == "all":
        check_downstream()
    print("ALL CHECKS PASSED")
