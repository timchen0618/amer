"""Run the full pipeline (raw + whitened) and print the decision.

    python run_all.py --data-dir demo --k 100 --m 3
"""
import argparse
import json
import os
import subprocess
import sys


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run([sys.executable] + cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--work-dir", default=None)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--corpus", default="corpus_emb.npy")
    ap.add_argument("--queries", default="q_single.npy")
    ap.add_argument("--no-faiss", action="store_true")
    ap.add_argument("--skip-whitening", action="store_true")
    a = ap.parse_args()

    d = a.data_dir
    w = a.work_dir or os.path.join(d, "analysis")
    os.makedirs(w, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    S = lambda n: os.path.join(here, n)  # noqa: E731
    nf = ["--no-faiss"] if a.no_faiss else []

    run([S("space_shape.py"), "--corpus", f"{d}/{a.corpus}", "--out", f"{w}/space_raw.json"])

    run(
        [S("rho.py"), "--corpus", f"{d}/{a.corpus}", "--golds", f"{d}/golds.jsonl",
         "--queries", f"{d}/{a.queries}", "--qids", f"{d}/qids.json",
         "--k", str(a.k), "--m", str(a.m), "--out", f"{w}/rho_raw.jsonl"] + nf
    )

    run(
        [S("predict_wins.py"), "--rho", f"{w}/rho_raw.jsonl",
         "--baseline", f"{d}/baseline_results.jsonl", "--system", f"{d}/amer_results.jsonl",
         "--out", f"{w}/wins_raw.json"]
    )

    if not a.skip_whitening:
        run(
            [S("whitening.py"), "--corpus", f"{d}/{a.corpus}",
             "--queries", f"{d}/{a.queries}", "--out-dir", f"{w}/whitened"]
        )
        run([S("space_shape.py"), "--corpus", f"{w}/whitened/{a.corpus}",
             "--out", f"{w}/space_white.json"])
        run(
            [S("rho.py"), "--corpus", f"{w}/whitened/{a.corpus}", "--golds", f"{d}/golds.jsonl",
             "--queries", f"{w}/whitened/{a.queries}", "--qids", f"{d}/qids.json",
             "--k", str(a.k), "--m", str(a.m), "--out", f"{w}/rho_white.jsonl"] + nf
        )

    # ---- decision
    def med(p):
        import numpy as np

        v = [json.loads(l)["rho_oracle"] for l in open(p)]
        return float(np.median(v)), float(np.mean([x > 2.0 for x in v]))

    r_med, r_hard = med(f"{w}/rho_raw.jsonl")
    print("\n" + "=" * 72)
    print(f"raw space       : median rho = {r_med:.2f},  frac(rho>2) = {r_hard:.1%}")
    if not a.skip_whitening:
        w_med, w_hard = med(f"{w}/rho_white.jsonl")
        print(f"whitened space  : median rho = {w_med:.2f},  frac(rho>2) = {w_hard:.1%}")
    print("=" * 72)
    print(
        """
Reading the outcome:

 (a) rho small before AND after whitening
     Single vectors are geometrically adequate on this benchmark. Your gains
     come from better centroid placement, not cluster coverage. -> reframe the
     paper around benchmark construction; find or build a high-rho dataset.

 (b) a meaningful fraction has rho > 2
     You have a per-example necessary condition for single-vector failure,
     computable before running any model. Report the AUC from predict_wins.py
     and make the criterion the contribution. -> the analysis-paper version.

 (c) rho small raw, large whitened
     Golds ARE separable but the encoder's geometry hides it. The binding
     constraint is document-side, not query count -- which means every
     query-side method (expansion, decomposition, MMR, MMLF, POQD, AMER) is
     pushing the wrong lever. Train the cached-embedding document adapter and
     show gains growing with rho. -> the strongest version of the paper.
"""
    )
    print(f"artifacts in {w}/")


if __name__ == "__main__":
    main()
