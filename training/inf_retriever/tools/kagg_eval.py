"""Score a multi-query checkpoint for every number of query embeddings k and every aggregation,
from one saved retrieval (retrieval_inf.py --save_per_step; code audit 2026-09-28, C2).

The first j generated embeddings do not depend on how many are generated in total (each step
re-encodes only its prefix), so the per-step lists of one max-k run give the results of every
k <= max k. Aggregation uses the same functions as retrieval_inf.py (round_robin, rrf).

    python training/inf_retriever/tools/kagg_eval.py --ds qampari \
        --per_step <dir>/dev_steps.jsonl --data data/phaseA/qampari_cleanv2dev500.jsonl \
        --ks 1 2 3 4 5 6 7 8 --aggs round_robin rrf --out <dir>/dev_kagg.json
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.getcwd())
from retrieval_inf import aggregate_round_robin, aggregate_rrf  # noqa: E402
from src.eval_utils import eval_retrieve_docs  # noqa: E402

AGG = {"round_robin": aggregate_round_robin, "rrf": aggregate_rrf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["qampari", "ambigqa"])
    ap.add_argument("--per_step", required=True)
    ap.add_argument("--data", required=True, help="eval jsonl the retrieval was run on (gold for scoring)")
    ap.add_argument("--ks", type=int, nargs="+", required=True)
    ap.add_argument("--aggs", nargs="+", default=["round_robin", "rrf"])
    ap.add_argument("--n_docs", type=int, default=500)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.per_step)]
    data = [json.loads(l) for l in open(args.data)]
    assert len(rows) == len(data)
    passages = {}
    for l in open(args.per_step + ".passages.jsonl"):
        p = json.loads(l)
        passages[p["id"]] = p
    kmax = len(rows[0]["lists"])
    assert max(args.ks) <= kmax, f"per-step file has only {kmax} embeddings per query"
    # all_results[step][query] = (ids, scores), as retrieval_inf.py builds it
    all_results = [[(r["lists"][j]["ids"], r["lists"][j]["scores"]) for r in rows] for j in range(kmax)]

    out = {"per_step": args.per_step, "data": args.data, "results": []}
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(args.out))) as tmp:
        for agg in args.aggs:
            for k in args.ks:
                merged = AGG[agg](all_results[:k], args.n_docs)
                path = os.path.join(tmp, f"{agg}_k{k}.jsonl")
                with open(path, "w") as f:
                    for ex, (ids, scores) in zip(data, merged):
                        ex = dict(ex)
                        ex["ctxs"] = [{"id": i, "title": passages[i]["title"], "text": passages[i]["text"],
                                       "score": str(float(sc))} for i, sc in zip(ids, scores)]
                        f.write(json.dumps(ex) + "\n")
                rec = {"agg": agg, "k": k}
                for topk in (100, 10):
                    r = eval_retrieve_docs(path, args.data, has_gold_id=(args.ds == "qampari"), topk=topk)
                    rec[f"mrecall@{topk}"], rec[f"recall@{topk}"] = float(r[0]), float(r[1])
                out["results"].append(rec)
                print(json.dumps(rec), flush=True)
                os.remove(path)
    json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
