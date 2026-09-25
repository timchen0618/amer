"""Steps 1-4 of the rho analysis, using EXISTING retrieved lists instead of a
fresh full-corpus k-NN search.

Why this works: r_k(q) -- the distance from the baseline's own query vector to
its k-th nearest corpus document -- is already sitting in the retrieval output
you already produced (results/finetuned/<dataset>/standard/<dataset>.jsonl):
ctxs is sorted by score and n_docs (8000) >> any k we care about. So r_k_query
and r_{k/m}_query are read off the existing scores, no FAISS, no full-corpus
array. See rho_analysis/README.md for what rho means and why.

This intentionally computes rho_query (query-centered), not rho_oracle
(gold-centroid-centered) -- the oracle variant needs a fresh k-NN search
centered on a point that was never actually queried, which this script does
not do. See README "Deferred" section.

Gold identity, per dataset (this is the part that needed real judgment calls
-- see the module docstring sections below and the final report):

  qampari         ground_truths is has_gold_id=True in the real eval pipeline
                  (scripts/eval/eval_qampari.sh). Each of the m clusters is a
                  paraphrase set for one gold entity; the representative id is
                  the cluster entry with the highest dataset-provided 'score'.

  ambigqa /       has_gold_id=False in the real eval pipeline
  ambigqa_2docs   (scripts/eval/eval_ambignq.sh) -- MRecall is computed by
                  text-matching answer aliases against retrieved passage text,
                  not by document identity. positive_ctxs has no 'id' field
                  and its has_answer flags are stale (computed against a
                  different candidate pool than this corpus). So there is no
                  ground-truth document id to look up. The representative for
                  aspect i is defined the same way MRecall itself is: the
                  highest-ranked doc in the BASELINE's own retrieved list
                  (top --gold-scan-depth, default 500) that satisfies
                  has_answer(aliases_i, text). Aspects with no satisfying doc
                  in that window are dropped for that example (reported as
                  gold_coverage). This makes D(q) and rho_query for these two
                  datasets a statement about "documents the baseline itself
                  can find", not the true unknown gold set -- flagged clearly
                  in the report, not silently treated as equivalent to qampari.

Usage:
    python pipeline_query.py --dataset qampari --k 100
    python pipeline_query.py --dataset ambigqa --k 100
    python pipeline_query.py --dataset ambigqa_2docs --k 100
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.eval_utils import SimpleTokenizer, eval_retrieve_docs, has_answer, read_jsonl  # noqa: E402

DATASETS = {
    "qampari": dict(
        eval_path="data/amer_data/eval_data/qampari.jsonl",
        standard_path="results/finetuned/qampari/standard/qampari.jsonl",
        system_path="results/finetuned/qampari/multi_hungarian/qampari.jsonl",
        shard_glob="wikipedia_embeddings/qampari/standard/passages_*",
        has_gold_id=True,
    ),
    "ambigqa": dict(
        eval_path="data/amer_data/eval_data/ambigqa.jsonl",
        standard_path="results/finetuned/ambigqa/standard/ambigqa.jsonl",
        system_path="results/finetuned/ambigqa/multi_hungarian/ambigqa.jsonl",
        shard_glob="wikipedia_embeddings/ambigqa/standard/passages_*",
        has_gold_id=False,
    ),
    "ambigqa_2docs": dict(
        eval_path="data/amer_data/eval_data/ambigqa_2docs.jsonl",
        standard_path="results/finetuned/ambigqa_2docs/standard/ambigqa_2docs.jsonl",
        system_path="results/finetuned/ambigqa_2docs/multi_hungarian/ambigqa_2docs.jsonl",
        shard_glob="wikipedia_embeddings/ambigqa_2docs/standard/passages_*",
        has_gold_id=False,
    ),
}


def qtext(row):
    return row.get("question_text", row.get("question"))


def resolve_gold_qampari(example):
    clusters = example.get("ground_truths") or example.get("positive_ctxs") or []
    reps = []
    for cluster in clusters:
        if not cluster:
            continue
        best = max(cluster, key=lambda d: d.get("score", 0))
        reps.append(best["id"])
    return reps


def resolve_gold_ambigqa(example, baseline_ctxs, tok, scan_depth):
    aspects = example.get("answers") or []
    reps = []
    window = baseline_ctxs[:scan_depth]
    for aliases in aspects:
        found = None
        for ctx in window:
            if has_answer(aliases, ctx["text"], tok):
                found = ctx["id"]
                break
        if found is not None:
            reps.append(found)
    return reps, len(aspects)


def sim_to_dist(sim):
    sim = max(min(float(sim), 1.0), -1.0)
    return float(np.sqrt(max(2.0 - 2.0 * sim, 0.0)))


def l2norm_row(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def pairwise_diameter(vecs):
    if len(vecs) < 2:
        return 0.0
    V = np.stack([l2norm_row(v) for v in vecs])
    G = V @ V.T
    d2 = np.maximum(2.0 - 2.0 * G, 0.0)
    return float(np.sqrt(d2.max()))


def scan_shards_for_ids(shard_glob, needed_ids):
    needed = set(needed_ids)
    found = {}
    files = sorted(glob.glob(os.path.join(ROOT, shard_glob)))
    if not files:
        raise FileNotFoundError(f"no shards matched {shard_glob}")
    for fp in files:
        remaining = needed - found.keys()
        if not remaining:
            break
        print(f"  scanning {os.path.basename(fp)} ({len(remaining)} ids still needed)", flush=True)
        with open(fp, "rb") as f:
            ids, embs = pickle.load(f)
        for i, _id in enumerate(ids):
            if _id in remaining:
                found[_id] = np.asarray(embs[i], dtype=np.float32)
        del ids, embs
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--gold-scan-depth", type=int, default=500)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--ks",
        default="5,10,20,50,100",
        help="comma-separated list of k values to additionally report rho_query/r_k_query at "
        "(rho is not scale-free -- r_k grows with k, so rho shrinks; report the full curve, "
        "not just one k). Always includes --k.",
    )
    a = ap.parse_args()
    ks = sorted(set(int(x) for x in a.ks.split(",")) | {a.k})

    cfg = DATASETS[a.dataset]
    out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", a.dataset)
    os.makedirs(out_dir, exist_ok=True)

    eval_path = os.path.join(ROOT, cfg["eval_path"])
    standard_path = os.path.join(ROOT, cfg["standard_path"])
    system_path = os.path.join(ROOT, cfg["system_path"])

    print(f"[{a.dataset}] loading eval data + baseline retrieval output", flush=True)
    eval_rows = read_jsonl(eval_path)
    baseline_rows = read_jsonl(standard_path)
    assert len(eval_rows) == len(baseline_rows), (len(eval_rows), len(baseline_rows))

    tok = SimpleTokenizer() if not cfg["has_gold_id"] else None

    # ---- 1. resolve gold representatives per example
    per_example_gold_ids = []
    n_aspects_total = []
    n_aspects_resolved = []
    for ex, base in zip(eval_rows, baseline_rows):
        if cfg["has_gold_id"]:
            gids = resolve_gold_qampari(ex)
            per_example_gold_ids.append(gids)
            n_aspects_total.append(len(ex.get("ground_truths") or []))
            n_aspects_resolved.append(len(gids))
        else:
            gids, n_asp = resolve_gold_ambigqa(ex, base["ctxs"], tok, a.gold_scan_depth)
            per_example_gold_ids.append(gids)
            n_aspects_total.append(n_asp)
            n_aspects_resolved.append(len(gids))

    needed_ids = set()
    for gids in per_example_gold_ids:
        needed_ids.update(gids)
    print(f"[{a.dataset}] {len(needed_ids)} distinct gold ids to look up across {len(eval_rows)} examples", flush=True)

    # ---- 2. pull gold embeddings from the standard-mode corpus shards
    id2emb = scan_shards_for_ids(cfg["shard_glob"], needed_ids)
    missing = needed_ids - id2emb.keys()
    if missing:
        print(f"[{a.dataset}] WARNING: {len(missing)}/{len(needed_ids)} gold ids not found in any shard", flush=True)

    # ---- 3. D(q), r_k_query, r_{k/m}_query, rho_query
    rho_rows = []
    for ex, base, gids, n_tot, n_res in zip(
        eval_rows, baseline_rows, per_example_gold_ids, n_aspects_total, n_aspects_resolved
    ):
        qid = qtext(ex)
        vecs = [id2emb[g] for g in gids if g in id2emb]
        m = max(len(vecs), 1)
        diam = pairwise_diameter(vecs)

        ctxs = base["ctxs"]

        by_k = {}
        for kk in ks:
            k_eff = min(kk, len(ctxs))
            km_eff = max(1, min(kk // m, len(ctxs)))
            r_k = sim_to_dist(ctxs[k_eff - 1]["score"])
            r_km = sim_to_dist(ctxs[km_eff - 1]["score"])
            by_k[str(kk)] = {
                "r_k_query": r_k,
                "r_k_over_m_query": r_km,
                "rho_query": float(diam / max(r_k, 1e-9)),
                "budget_shrink": float(r_k / max(r_km, 1e-9)),
            }

        primary = by_k[str(a.k)]
        rho_rows.append(
            {
                "qid": qid,
                "n_aspects_total": int(n_tot),
                "n_aspects_resolved": int(n_res),
                "gold_coverage": float(n_res / n_tot) if n_tot else None,
                "n_gold_embedded": len(vecs),
                "diameter": diam,
                "m_used": m,
                "r_k_query": primary["r_k_query"],
                "r_k_over_m_query": primary["r_k_over_m_query"],
                "rho_query": primary["rho_query"],
                "budget_shrink": primary["budget_shrink"],
                "by_k": by_k,
            }
        )

    rho_path = os.path.join(out_dir, "rho_query.jsonl")
    with open(rho_path, "w") as f:
        for r in rho_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[{a.dataset}] wrote {rho_path} (n={len(rho_rows)})", flush=True)

    # ---- 4. per-example MRecall for baseline (standard) and system (multi_hungarian)
    for name, path in (("baseline", standard_path), ("system", system_path)):
        result = eval_retrieve_docs(path, eval_path, has_gold_id=cfg["has_gold_id"], topk=a.k)
        mrecall_list = result[-2]
        qids = [qtext(r) for r in eval_rows]
        assert len(qids) == len(mrecall_list), (len(qids), len(mrecall_list))
        out_path = os.path.join(out_dir, f"{name}_results.jsonl")
        with open(out_path, "w") as f:
            for qid, mr in zip(qids, mrecall_list):
                f.write(json.dumps({"qid": qid, "mrecall": float(mr)}) + "\n")
        print(f"[{a.dataset}] wrote {out_path}", flush=True)

    # ---- coverage summary (only meaningful for ambigqa/ambigqa_2docs)
    cov = [r["gold_coverage"] for r in rho_rows if r["gold_coverage"] is not None]
    if cov:
        print(
            f"[{a.dataset}] gold aspect coverage within top-{a.gold_scan_depth}: "
            f"mean={np.mean(cov):.3f} median={np.median(cov):.3f} "
            f"frac_zero_coverage={np.mean([c == 0 for c in cov]):.3f}",
            flush=True,
        )

    print(f"[{a.dataset}] done.", flush=True)


if __name__ == "__main__":
    main()
