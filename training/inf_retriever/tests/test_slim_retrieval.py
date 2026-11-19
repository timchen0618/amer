"""End-to-end test of tools/slim_retrieval.py on real full-eval folders (one QAMPARI, one AmbigQA; dev and test).

For each <folder>/<split>/<ds>.jsonl (the original is only read, never modified), in a scratch directory:
  1. eval.py on the original reproduces the recorded <split>/eval_metrics.txt (MRecall lines at top-100 and top-10);
  2. slim -> verify (same rows, query identity, ctx ids and scores in order) and report the size ratio;
  3. eval.py on the slim file: QAMPARI (gold-id scoring) must reproduce the recorded metrics; AmbigQA (answer-string
     matching on passage text) must fail, which is why AmbigQA needs `restore`;
  4. restore --data <eval data> (title/text from the corpus) gives back the original file record for record, and
     byte for byte;
  5. eval.py on the restored file reproduces the recorded metrics;
  6. QAMPARI: tools/fusion_combo.py:load_single returns the same ids and scores from the slim and original files.

Needs pytrec_eval (div env) and a compute node (reads the 15 GB corpus):
  srun --account=torch_pr_152_echoi --time=01:30:00 --mem=48G --cpus-per-task=4 singularity exec \
    --overlay /scratch/hc3337/envs/div.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; \
             python training/inf_retriever/tests/test_slim_retrieval.py --work <scratch dir>"
"""
import argparse, filecmp, json, os, re, subprocess, sys

ROOT = "/scratch/hc3337/projects/autoregressive"
sys.path.insert(0, f"{ROOT}/training/inf_retriever/tools")
import slim_retrieval as S  # noqa: E402

CASES = [
    ("qampari", "results/phaseC/full/qampari/clean_v3_f_qampari_S0_single_10k_2step_causal_t002_s9000"),
    ("ambigqa", "results/phaseC/full/ambigqa/clean_v3_abl_ambigqa_single_s150"),
]


def recorded(metrics):
    txt = open(metrics).read()
    data = re.search(r"^Data path: (.*)$", txt, re.M).group(1).strip()
    gold = re.search(r"^Has gold ID:\s+(\w+)", txt, re.M).group(1) == "True"
    return data, gold, re.findall(r"^MRecall: .*$", txt, re.M)


def run_eval(inp, data, gold):
    cmd = [sys.executable, "eval.py", "--data_path", data, "--topk", "100", "10", "--input-file", inp]
    if not gold:
        cmd.append("--no-gold-id")
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return p.returncode, re.findall(r"^MRecall: .*$", p.stdout, re.M), p.stderr[-400:]


def check(ds, folder, split, work):
    tag = f"{ds}/{os.path.basename(folder)}/{split}"
    orig = f"{ROOT}/{folder}/{split}/{ds}.jsonl"
    data, gold, rec = recorded(f"{ROOT}/{folder}/{split}/eval_metrics.txt")
    assert len(rec) == 2, (tag, rec)
    before = os.stat(orig)

    rc, got, err = run_eval(orig, data, gold)
    assert rc == 0 and got == rec, (tag, "original does not reproduce the record", got, rec, err)
    print(f"[{tag}] 1. original reproduces record: {rec}")

    sl = f"{work}/{ds}_{split}.slim.jsonl"
    S.slim(orig, sl)
    n = S.verify(orig, sl)
    print(f"[{tag}] 2. slim verified, {n} rows, {os.path.getsize(orig) / 2**20:.0f} MB -> "
          f"{os.path.getsize(sl) / 2**20:.0f} MB ({100 * os.path.getsize(sl) / os.path.getsize(orig):.1f}%)")

    rc, got, err = run_eval(sl, data, gold)
    if gold:
        assert rc == 0 and got == rec, (tag, "slim does not reproduce the record", got, rec, err)
        print(f"[{tag}] 3. eval on slim reproduces record")
    else:
        assert rc != 0 and "'text'" in err, (tag, "expected eval on slim to fail on missing text", rc, got, err)
        print(f"[{tag}] 3. eval on slim fails as expected (needs passage text): {err.strip().splitlines()[-1]}")

    rs = f"{work}/{ds}_{split}.restored.jsonl"
    S.restore(sl, rs, data=f"{ROOT}/{data}")
    for i, (a, b) in enumerate(zip(S.read(orig), S.read(rs))):
        assert a == b and list(a) == list(b), (tag, f"restored row {i} differs")
    same_bytes = filecmp.cmp(orig, rs, shallow=False)
    assert same_bytes, (tag, "restored file is not byte-identical")
    print(f"[{tag}] 4. restored file identical to original (records and bytes)")

    rc, got, err = run_eval(rs, data, gold)
    assert rc == 0 and got == rec, (tag, "restored does not reproduce the record", got, rec, err)
    print(f"[{tag}] 5. eval on restored reproduces record")

    if ds == "qampari":
        import fusion_combo as F
        os.makedirs(f"{work}/fc/{split}", exist_ok=True)
        link = f"{work}/fc/{split}/qampari.jsonl"
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(sl, link)
        assert F.load_single(f"{work}/fc", split) == F.load_single(f"{ROOT}/{folder}", split), tag
        print(f"[{tag}] 6. fusion_combo.load_single identical on slim")

    after = os.stat(orig)
    assert (before.st_size, before.st_mtime) == (after.st_size, after.st_mtime), (tag, "original was modified")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    for ds, folder in CASES:
        for split in ("test", "dev"):
            check(ds, folder, split, a.work)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
