"""Inference-side fusion study and paper-subset scores (objective audit, 2026-10-15). CPU only, no model.

For each checkpoint directory (results/phaseC/full/qampari/<run>_s<step>) it reads the saved per-query lists
({dev,test}_steps.jsonl: k lists x top-500 ids + scores, from retrieval_inf.py --save_per_step) and the query
vectors ({dev,test}_qemb.npy), fuses the k lists with every method below, and scores MRecall@100 / Recall@100 on
  - full dev (data/phaseA/qampari_cleanv2dev500.jsonl),
  - full test (data/amer_data/eval_data/qampari.jsonl),
  - the paper's large- / small-distance thirds of test (data/qampari/test531_{large,small}_distance_indices_inf.txt:
    data/qampari/{large,small}_distance_indices_inf.txt, which index the paper's 1000-question QAMPARI file,
    mapped by question text onto the 531 test questions; 168 large, 180 small; test only, never used to select).
Scoring is eval_utils.eval_retrieve_docs with gold ids, re-implemented on ids: a gold cluster counts as found
when any of its ids is in the top 100; MRecall = every cluster found (or >= 100 found); Recall = share found.
`rrf_c60` reproduces the headline (retrieval_inf.aggregate_rrf, k = 5); check it against the logged results.

Methods (query 1 = the first generated embedding = output 0):
  rrf_c60                  plain RRF, sum_j 1 / (60 + rank_j(d))                         (the headline)
  rr                       round robin (retrieval_inf.aggregate_round_robin)
  lead1_N{N}               query 1's top N, then round robin over queries 2..k, skipping duplicates
  wrrf_w{w}_c{c}           sum_j w_j / (c + rank_j(d)); w_1 = w, others 1
  max_score                a passage's score = its best score over the k lists
  sum_score                sum over lists of its score; a list it is missing from contributes that list's 500th score
  dedup{t}_rrf             drop query j if cos(q_j, an earlier kept query) > t, then RRF (c = 60) over the rest
  gate{t}                  if the mean pairwise cosine of the k queries > t, query 1 alone; else RRF (c = 60)
A single-query output (test/qampari.jsonl with ctxs) can be scored on the same subsets with --single.

    python training/inf_retriever/tools/fusion_eval.py --dirs <dir> [<dir> ...] --out <json>
    python training/inf_retriever/tools/fusion_eval.py --single <name>=<test jsonl> ... --out <json>
"""
import argparse, json, os
import numpy as np

DEV = "data/phaseA/qampari_cleanv2dev500.jsonl"
TEST = "data/amer_data/eval_data/qampari.jsonl"
SUB = {s: f"data/qampari/test531_{s}_distance_indices_inf.txt" for s in ("large", "small")}
# Dev thirds with the paper's thresholds (tools/dev_distance_subsets.py, 2026-10-23): 173 large / 165 small of 500.
DEV_SUB = {s: f"data/qampari/dev500_{s}_distance_indices_inf.txt" for s in ("large", "small")}
TOPK, NDOCS = 100, 500
LEAD_N = (0, 20, 40, 60, 80, 100)
WRRF = [(w, c) for w in (1, 1.5, 2, 3, 5) for c in (10, 60)]
DEDUP_T = (0.90, 0.93, 0.95, 0.97, 0.99)
GATE_T = (0.85, 0.90, 0.93, 0.95, 0.97)


def clusters(path):
    out = []
    for l in open(path):
        q = json.loads(l)
        g = q.get("positive_ctxs") or q.get("ground_truths")
        out.append([{str(d["id"]) for d in (c if isinstance(c, list) else [c])} for c in g])
    return out


def score(fused, golds, idx=None):
    """fused: list of id lists (one per question). Returns (MRecall@100, Recall@100) in %, over idx if given."""
    rng = range(len(golds)) if idx is None else idx
    mr, rc = [], []
    for i in rng:
        top = set(fused[i][:TOPK])
        hit = [bool(c & top) for c in golds[i]]
        n_top = min(TOPK, len(fused[i]))
        mr.append(sum(hit) >= n_top if len(hit) > n_top else all(hit))
        rc.append(sum(hit) / len(hit))
    return round(100 * float(np.mean(mr)), 2), round(100 * float(np.mean(rc)), 2)


def rrf(lists, w=None, c=60):
    s = {}
    for j, ids in enumerate(lists):
        wj = 1.0 if w is None else w[j]
        for r, d in enumerate(ids, 1):
            s[d] = s.get(d, 0.0) + wj / (c + r)
    return [d for d, _ in sorted(s.items(), key=lambda t: -t[1])][:NDOCS]


def round_robin(lists):
    out, seen = [], set()
    for r in range(max(map(len, lists))):
        for ids in lists:
            if r < len(ids) and ids[r] not in seen:
                seen.add(ids[r]); out.append(ids[r])
    return out[:NDOCS]


def lead1(lists, n):
    if n == 0:
        return round_robin(lists)
    out = list(lists[0][:n]); seen = set(out)
    for d in round_robin(lists[1:]) if len(lists) > 1 else lists[0][n:]:
        if d not in seen:
            seen.add(d); out.append(d)
    if len(out) < TOPK:  # only query 1 left to fill from
        out += [d for d in lists[0] if d not in seen]
    return out[:NDOCS]


def by_score(lists, scores, how):
    s = {}
    floor = [sc[-1] for sc in scores]
    for ids, sc in zip(lists, scores):
        for d, v in zip(ids, sc):
            s.setdefault(d, []).append(v)
    if how == "max":
        val = {d: max(v) for d, v in s.items()}
    else:  # sum, missing lists contribute their floor score
        k = len(lists)
        present = {d: set() for d in s}
        for j, ids in enumerate(lists):
            for d in ids:
                present[d].add(j)
        val = {d: sum(s[d]) + sum(floor[j] for j in range(k) if j not in present[d]) for d in s}
    return [d for d, _ in sorted(val.items(), key=lambda t: -t[1])][:NDOCS]


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def methods(lists, scores, q):
    """All fused lists for one question. q: (k, d) unit query vectors."""
    out = {"rrf_c60": rrf(lists), "rr": round_robin(lists)}
    for n in LEAD_N:
        out[f"lead1_N{n}"] = lead1(lists, n)
    k = len(lists)
    for w, c in WRRF:
        out[f"wrrf_w{w}_c{c}"] = rrf(lists, [w] + [1.0] * (k - 1), c)
    out["max_score"] = by_score(lists, scores, "max")
    out["sum_score"] = by_score(lists, scores, "sum")
    cos = q @ q.T
    for t in DEDUP_T:
        keep = []
        for j in range(k):
            if all(cos[j, i] <= t for i in keep):
                keep.append(j)
        out[f"dedup{t}_rrf"] = rrf([lists[j] for j in keep])
    mean_qq = (cos.sum() - np.trace(cos)) / (k * (k - 1)) if k > 1 else 1.0
    for t in GATE_T:
        out[f"gate{t}"] = list(lists[0][:NDOCS]) if mean_qq > t else out["rrf_c60"]
    return out


def load_sub():
    out = {s: [int(x) for x in open(p).read().split()] for s, p in SUB.items()}
    if all(os.path.exists(p) for p in DEV_SUB.values()):
        out.update({f"dev_{s}": [int(x) for x in open(p).read().split()] for s, p in DEV_SUB.items()})
    return out


def eval_dir(d, golds, sub):
    res = {}
    for split in ("dev", "test"):
        if not os.path.exists(f"{d}/{split}_steps.jsonl"):  # pruned steps have dev lists only
            continue
        rows = [json.loads(l) for l in open(f"{d}/{split}_steps.jsonl")]
        Q = np.load(f"{d}/{split}_qemb.npy").astype(np.float32)
        k = len(rows[0]["lists"])
        Q = unit(Q.reshape(len(rows), k, -1))
        assert len(rows) == len(golds[split])
        fused = {}
        for i, r in enumerate(rows):
            m = methods([l["ids"] for l in r["lists"]], [l["scores"] for l in r["lists"]], Q[i])
            for name, ids in m.items():
                fused.setdefault(name, []).append(ids)
        for name, f in fused.items():
            e = res.setdefault(name, {})
            e[split] = score(f, golds[split])
            for s, idx in sub.items():
                if split == "test" and not s.startswith("dev_"):
                    e[f"test_{s}"] = score(f, golds[split], idx)
                elif split == "dev" and s.startswith("dev_"):
                    e[s] = score(f, golds[split], idx)
        if split == "test":
            res["_q1_only"] = {"test": score([r["lists"][0]["ids"] for r in rows], golds["test"])}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--single", nargs="*", default=[], help="name=path to a test retrieval jsonl with ctxs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dev", default=DEV, help="full-dev file the dirs were scored on (default: the old 500-question set)")
    a = ap.parse_args()
    golds = {"dev": clusters(a.dev), "test": clusters(TEST)}
    sub = load_sub()
    if a.dev != DEV:  # the dev distance thirds index the old 500-question set only
        sub = {k: v for k, v in sub.items() if not k.startswith("dev_")}
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for spec in a.single:
        name, path = spec.split("=", 1)
        ids = [[str(c["id"]) for c in json.loads(l)["ctxs"]] for l in open(path)]
        e = {"test": score(ids, golds["test"])}
        for s, idx in [(s, x) for s, x in sub.items() if not s.startswith('dev_')]:
            e[f"test_{s}"] = score(ids, golds["test"], idx)
        out[name] = {"single": e}
        print(name, json.dumps(e), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    for d in a.dirs:
        name = os.path.basename(d.rstrip("/"))
        out[name] = eval_dir(d, golds, sub)
        print(name, "rrf_c60", json.dumps(out[name]["rrf_c60"]), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
