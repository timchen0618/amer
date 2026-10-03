"""Phase A: build reduced retrieval corpora for cheap full-pipeline evaluation.

For each query set, the reduced corpus is
    union over queries of the BASE retriever's top-`topk` passages
  + the gold passages (QAMPARI: `ground_truths` ids; AmbigQA is scored by answer strings)
  + `n_random` random corpus passages (shared distractors, seeded),
written as a TSV with the same header as the full corpus (id, text, title), so
gen_embed_new.py / retrieval_inf.py / retrieval_base.py can use it unchanged.
One pass over the 15 GB corpus serves every query set.

Usage (repo root, compute node, ~64 GB RAM):
    python training/inf_retriever/tools/phaseA_build_reduced_corpus.py \
        --spec qampari_test:results/base_retrievers/inf/amer_data/qampari_5_to_8_ctxs.jsonl:data/amer_data/eval_data/qampari.jsonl \
        --spec ambigqa_test:results/base_retrievers/inf/amer_data/ambigqa.jsonl: \
        --topk 1000 --n_random 100000 --out_dir data/phaseA/corpora
Each --spec is NAME:BASE_RESULTS_JSONL:EVAL_JSONL_FOR_GOLD_IDS (last field may be empty).
"""
import argparse
import json
import os
import random
import sys

CORPUS = "/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv"


def base_candidates(path, topk):
    ids, n = set(), 0
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            ids.update(c["id"] for c in ex["ctxs"][:topk])
            n += 1
    return ids, n


def gold_ids(path):
    ids = set()
    if not path:
        return ids
    with open(path) as f:
        for line in f:
            for group in json.loads(line).get("ground_truths", []):
                ids.update(g["id"] for g in group)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", required=True)
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--n_random", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="data/phaseA/corpora")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Shared random distractors: reservoir-free sampling by line index.
    with open(CORPUS) as f:
        n_lines = sum(1 for _ in f) - 1  # minus header
    rand_idx = set(random.Random(args.seed).sample(range(n_lines), args.n_random))
    print(f"corpus passages: {n_lines}; random distractors: {len(rand_idx)}", flush=True)

    specs = []
    for s in args.spec:
        name, base, gold = (s.split(":") + [""])[:3]
        cand, nq = base_candidates(base, args.topk)
        g = gold_ids(gold)
        specs.append({"name": name, "ids": cand | g, "n_base": len(cand), "n_gold": len(g), "nq": nq})
        print(f"{name}: {nq} queries, {len(cand)} base top-{args.topk} passages, {len(g)} gold ids "
              f"({len(g - cand)} not already among base candidates)", flush=True)

    outs = {s["name"]: open(os.path.join(args.out_dir, f"{s['name']}.tsv"), "w") for s in specs}
    counts = {s["name"]: 0 for s in specs}
    found = {s["name"]: set() for s in specs}
    with open(CORPUS) as f:
        header = f.readline()
        for o in outs.values():
            o.write(header)
        for i, line in enumerate(f):
            pid = line.split("\t", 1)[0]
            is_rand = i in rand_idx
            for s in specs:
                if is_rand or pid in s["ids"]:
                    outs[s["name"]].write(line)
                    counts[s["name"]] += 1
                    if pid in s["ids"]:
                        found[s["name"]].add(pid)
    for s in specs:
        outs[s["name"]].close()
        missing = len(s["ids"] - found[s["name"]])
        print(f"{s['name']}: wrote {counts[s['name']]} passages "
              f"({len(found[s['name']])} candidates/golds + random; {missing} candidate ids not in corpus)", flush=True)
    # Merge into build_info.json (one record per set, with the settings it was built with), so
    # building a new set does not erase the record of the existing ones.
    info_path = os.path.join(args.out_dir, "build_info.json")
    info = json.load(open(info_path)) if os.path.exists(info_path) else {}
    sets = info.get("sets", {})
    for name in sets:  # older records stored the settings once at the top level
        for k in ("topk", "n_random", "seed"):
            if k in info:
                sets[name].setdefault(k, info[k])
    for s in specs:
        sets[s["name"]] = {"queries": s["nq"], "base_candidates": s["n_base"], "gold_ids": s["n_gold"],
                           "passages": counts[s["name"]], "topk": args.topk, "n_random": args.n_random,
                           "seed": args.seed, "spec": next(x for x in args.spec if x.split(":")[0] == s["name"])}
    with open(info_path, "w") as f:
        json.dump({"sets": sets}, f, indent=1)


if __name__ == "__main__":
    main()
