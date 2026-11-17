"""Slim every scored retrieval output under results/phaseC/full in place (see slim_retrieval.py), one shard of the files.

Per file <run>_s<step>/<split>/<ds>.jsonl, skipped unless <split>/eval_metrics.txt has an MRecall line, the folder
was last modified more than --min_age_h hours ago and the run is not in --skip_runs (runs still training or being
evaluated). The original is replaced only after:
  (a) every ctx title/text in it equals the corpus row of its id (so slim_retrieval.py restore gives it back), and
  (b) the slim copy has the same rows, query identity and ctx ids/scores in order (slim_retrieval.verify).
Any failure leaves the original untouched and is logged as FAIL.

    python slim_all.py --shard 0 --nshards 16 [--skip_runs RUN ...] [--dry_run]
    python slim_all.py --shard 0 --nshards 1 --runs RUN --min_age_h 0   # one finished run (f_run.sh, after RUN DONE)
"""
import argparse, glob, os, re, sys, time, zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slim_retrieval as S  # noqa: E402

FULL = "/scratch/hc3337/projects/autoregressive/results/phaseC/full"


def candidates(min_age_h, skip_runs, runs=None):
    out = []
    now = time.time()
    for ds in ("qampari", "ambigqa"):
        for f in sorted(glob.glob(f"{FULL}/{ds}/*/*/{ds}.jsonl")):
            split_dir = os.path.dirname(f)
            folder = os.path.dirname(split_dir)
            run = re.sub(r"_s\d+$", "", os.path.basename(folder))
            if run in skip_runs or (runs and run not in runs) or os.path.basename(folder).startswith("."):
                continue
            try:
                if "MRecall" not in open(f"{split_dir}/eval_metrics.txt").read():
                    continue
            except OSError:
                continue
            if now - os.path.getmtime(f) < min_age_h * 3600 or now - os.path.getmtime(folder) < min_age_h * 3600:
                continue
            out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--skip_runs", nargs="*", default=[])
    ap.add_argument("--runs", nargs="*", default=[], help="only these runs (default: all)")
    ap.add_argument("--min_age_h", type=float, default=6)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    # Stable assignment by path hash (not list position), so each file belongs to exactly one shard whatever the
    # other shards have already slimmed or whenever this shard starts.
    files = [f for f in candidates(a.min_age_h, set(a.skip_runs), set(a.runs))
             if zlib.crc32(f.encode()) % a.nshards == a.shard and not S.is_slim(f)]
    log = lambda m: print(f"[{time.strftime('%F %T')}] [shard {a.shard}] {m}", flush=True)
    log(f"{len(files)} files, {sum(os.path.getsize(f) for f in files) / 2**30:.1f} GB")
    if a.dry_run:
        for f in files:
            print(f)
        return
    ids = set()
    for f in files:
        ids |= S.ctx_ids(f)
    log(f"{len(ids)} distinct passages; loading them from the corpus")
    passages = S.load_corpus(ids, strict=False)  # a file citing a missing id FAILs alone in restore_check
    del ids
    before = after = 0
    for f in files:
        tmp = f"{f}.slim{a.shard}.{os.getpid()}.tmp"  # pid: a driver and a bulk run never share a temp file
        try:
            S.restore_check(f, passages)
            S.slim(f, tmp)
            n = S.verify(f, tmp)
            b, s = os.path.getsize(f), os.path.getsize(tmp)
            os.replace(tmp, f)
            before += b; after += s
            log(f"OK {n} rows {b / 2**20:.0f} MB -> {s / 2**20:.0f} MB {f}")
        except Exception as e:  # leave the original as it was
            if os.path.exists(tmp):
                os.remove(tmp)
            log(f"FAIL {f}: {type(e).__name__}: {str(e)[:300]}")
    log(f"DONE {before / 2**30:.1f} GB -> {after / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
