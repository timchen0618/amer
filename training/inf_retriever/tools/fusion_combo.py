"""Fusion combinations (objective audit, 2026-10-16), follow-up to fusion_eval.py. CPU only, no model.

Same inputs and scoring as fusion_eval.py (per-query lists of each multi-query checkpoint, full dev / test, the
paper's large / small distance thirds of test), plus the single-query model's own top-500 list of the same setting
(--single_dir: <dir>/{dev,test}/qampari.jsonl with ctxs + scores). Methods (query 1 = output 0; s = the single
model's list):
  single                   s alone (reference)
  rrf_c60                  plain RRF over the k multi lists (the headline, reference)
  wsum_w{w}                1. weighted summed scores: w * score_1 + sum_{j>1} score_j (a missing list
                              contributes its 500th score, as fusion_eval sum_score; w = 1 is sum_score)
  hyb_rrf_ws{w}            2. hybrid: RRF (c = 60) over [s (weight w)] + multi queries 2..k
  hyb5_rrf_ws{w}           2b. RRF over [s (weight w)] + all k multi queries
  hyb_sum_ws{w}            2c. summed scores over [s (weight w)] + multi queries 2..k
  gate{t}_hyb_rrf_ws{w}    3. s alone when the multi queries agree (mean q-q cos > t), else hyb_rrf_ws{w}
Caveat: every hyb* method is a two-model ensemble (the single model + a multi model), so it costs two encoders.
Control (--partners): the same single model fused with ANOTHER single-query model instead, ens_r{r} = RRF (c = 60)
over [s (weight r), partner (weight 1)], r chosen on dev like the hybrids; if it gains as much, the hybrid's gain
is ensembling, not multiple queries.
Combination 4 (per-family rule) is computed from fusion_eval's json by fusion_combo_summary.py.

    python training/inf_retriever/tools/fusion_combo.py --single_dir <dir> --dirs <dir> [<dir> ...] --out <json>
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fusion_eval import DEV, TEST, NDOCS, clusters, score, rrf, unit, load_sub  # noqa: E402

WS = (1, 1.5, 2, 3)
GATE_T = (0.85, 0.90, 0.93, 0.95, 0.97)


def wsum(lists, scores, w):
    """sum_j w_j * score_j(d); a list d is missing from contributes w_j * its last (500th) score."""
    s = {}
    for j, (ids, sc) in enumerate(zip(lists, scores)):
        for d, v in zip(ids, sc):
            s.setdefault(d, {})[j] = v
    floor = [sc[-1] for sc in scores]
    val = {d: sum(w[j] * p.get(j, floor[j]) for j in range(len(lists))) for d, p in s.items()}
    return [d for d, _ in sorted(val.items(), key=lambda t: -t[1])][:NDOCS]


def methods(lists, scores, q, s_ids, s_sc):
    k = len(lists)
    out = {"single": s_ids[:NDOCS], "rrf_c60": rrf(lists)}
    for w in WS:
        out[f"wsum_w{w}"] = wsum(lists, scores, [w] + [1.0] * (k - 1))
        out[f"hyb_rrf_ws{w}"] = rrf([s_ids] + lists[1:], [w] + [1.0] * (k - 1))
        out[f"hyb5_rrf_ws{w}"] = rrf([s_ids] + lists, [w] + [1.0] * k)
        out[f"hyb_sum_ws{w}"] = wsum([s_ids] + lists[1:], [s_sc] + scores[1:], [w] + [1.0] * (k - 1))
    cos = q @ q.T
    mean_qq = (cos.sum() - np.trace(cos)) / (k * (k - 1))
    for t in GATE_T:
        for w in WS:
            out[f"gate{t}_hyb_rrf_ws{w}"] = out["single"] if mean_qq > t else out[f"hyb_rrf_ws{w}"]
    return out


def load_single(d, split):
    rows = [json.loads(l) for l in open(f"{d}/{split}/qampari.jsonl")]
    return [[str(c["id"]) for c in r["ctxs"]] for r in rows], [[float(c["score"]) for c in r["ctxs"]] for r in rows]


def eval_dir(d, golds, sub, single):
    res = {}
    for split in ("dev", "test"):
        rows = [json.loads(l) for l in open(f"{d}/{split}_steps.jsonl")]
        k = len(rows[0]["lists"])
        Q = unit(np.load(f"{d}/{split}_qemb.npy").astype(np.float32).reshape(len(rows), k, -1))
        s_ids, s_sc = single[split]
        assert len(rows) == len(golds[split]) == len(s_ids)
        fused = {}
        for i, r in enumerate(rows):
            m = methods([l["ids"] for l in r["lists"]], [l["scores"] for l in r["lists"]], Q[i], s_ids[i], s_sc[i])
            for name, ids in m.items():
                fused.setdefault(name, []).append(ids)
        for name, f in fused.items():
            e = res.setdefault(name, {})
            e[split] = score(f, golds[split])
            if split == "test":
                for s, idx in sub.items():
                    e[f"test_{s}"] = score(f, golds[split], idx)
    return res


ENS_R = (0.5, 0.75, 1, 1.5, 2, 3, 5)


def eval_partner(d, golds, sub, single):
    res = {}
    for split in ("dev", "test"):
        p_ids, _ = load_single(d, split)
        s_ids, _ = single[split]
        assert len(p_ids) == len(s_ids) == len(golds[split])
        for r in ENS_R:
            f = [rrf([a, b], [r, 1.0]) for a, b in zip(s_ids, p_ids)]
            e = res.setdefault(f"ens_r{r}", {})
            e[split] = score(f, golds[split])
            if split == "test":
                for s_, idx in sub.items():
                    e[f"test_{s_}"] = score(f, golds[split], idx)
        e = res.setdefault("partner_alone", {})
        e[split] = score(p_ids, golds[split])
        if split == "test":
            for s_, idx in sub.items():
                e[f"test_{s_}"] = score(p_ids, golds[split], idx)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--single_dir", required=True)
    ap.add_argument("--dirs", nargs="*", default=[])
    ap.add_argument("--partners", nargs="*", default=[], help="other single-query result dirs ({dev,test}/qampari.jsonl)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    golds = {"dev": clusters(DEV), "test": clusters(TEST)}
    sub = load_sub()
    single = {sp: load_single(a.single_dir, sp) for sp in ("dev", "test")}
    out = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for d in a.dirs:
        name = os.path.basename(d.rstrip("/"))
        out[name] = eval_dir(d, golds, sub, single)
        print(name, "single", json.dumps(out[name]["single"]), "| rrf", json.dumps(out[name]["rrf_c60"]), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)
    for d in a.partners:
        name = "partner:" + os.path.basename(d.rstrip("/"))
        out[name] = eval_partner(d, golds, sub, single)
        print(name, json.dumps(out[name]), flush=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
