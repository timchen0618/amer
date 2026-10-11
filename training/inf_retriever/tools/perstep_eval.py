"""Score each query embedding of a multi-query checkpoint on its own (MRecall@100 / Recall@100 of
embedding j's top-100 alone), from the per-step lists of a full eval (retrieval_inf.py --save_per_step).

QAMPARI is scored by gold id, exactly as src/eval_utils.eval_retrieve_docs(has_gold_id=True) does:
a gold cluster is found if any of its ids is in the top 100; MRecall = all clusters found (or at least
100 when a question has more). Embedding 1 alone equals the RRF k = 1 score in *_kagg.json, which is
checked for every file.

    python training/inf_retriever/tools/perstep_eval.py results/phaseC/full/qampari/<tag> [...]
writes <tag>/{dev,test}_perstep.json
"""
import json, sys

DATA = {"dev": "data/phaseA/qampari_cleanv2dev500.jsonl", "test": "data/amer_data/eval_data/qampari.jsonl"}
TOPK = 100


def gold_clusters(inst):
    cl = inst.get("positive_ctxs") or inst.get("ground_truths")
    return [{d["id"] for d in c} for c in cl]


def score(ids_per_q, golds):
    mr, rc = [], []
    for ids, clusters in zip(ids_per_q, golds):
        top = set(ids[:TOPK])
        hit = [bool(c & top) for c in clusters]
        need = min(len(clusters), min(TOPK, len(ids)))
        mr.append(int(sum(hit) >= need))
        rc.append(sum(hit) / len(hit))
    return 100 * sum(mr) / len(mr), 100 * sum(rc) / len(rc)


def main():
    golds = {s: [gold_clusters(json.loads(l)) for l in open(p)] for s, p in DATA.items()}
    for d in sys.argv[1:]:
        for split in ("dev", "test"):
            try:
                rows = [json.loads(l) for l in open(f"{d}/{split}_steps.jsonl")]
            except OSError:
                continue
            assert len(rows) == len(golds[split])
            k = len(rows[0]["lists"])
            res = []
            for j in range(k):
                m, r = score([row["lists"][j]["ids"] for row in rows], golds[split])
                res.append({"embedding": j + 1, "mrecall@100": round(m, 2), "recall@100": round(r, 2)})
            try:  # embedding 1 alone must equal RRF k = 1
                ka = {(x["agg"], x["k"]): x["mrecall@100"] for x in json.load(open(f"{d}/{split}_kagg.json"))["results"]}
                assert abs(ka[("rrf", 1)] - res[0]["mrecall@100"]) < 0.01, (d, split, ka[("rrf", 1)], res[0])
            except OSError:
                pass
            json.dump({"per_step": f"{d}/{split}_steps.jsonl", "results": res}, open(f"{d}/{split}_perstep.json", "w"), indent=1)
            print(d.rsplit("/", 1)[-1], split, [x["mrecall@100"] for x in res], flush=True)


if __name__ == "__main__":
    main()
