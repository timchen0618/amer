"""Copy statistics of a saved multi-query checkpoint, recomputed offline (objective audit, 2026-10-19).

The training log records copy_stats (src/inbatch.py) only for runs trained after 2026-10-13; older runs such as
clean_v3_abl2_qampari_C_multi have none. This runs training batches through a checkpoint with no gradient and
every step teacher-forced (sampling_rate 0), and averages the same statistics the log records:
  tf_copy_cos        cos(output t, gold t-1), the gold just fed in;
  tf_best_unfed_cos  the largest cos(output t, gold) over golds not fed in before output t;
  tf_copy_match      the share of outputs t the loss matches to gold t-1.
Batches come from the run's own training data and sampler (prepare_data with the checkpoint's saved opt).

    python training/inf_retriever/tools/copy_stats_offline.py --ckpt <dir with checkpoint.pth> [--batches 40]
"""
import argparse, importlib.util, json, os, sys
import torch

REPO = "/scratch/hc3337/projects/autoregressive"
sys.path.insert(0, REPO + "/training/inf_retriever")
sys.path.append(REPO)  # for src/inference_utils.py's `training.inf_retriever.src` import; appended so `src` stays the training one
os.chdir(REPO)
import finetuning_multi  # noqa: E402  (also makes `src` = training/inf_retriever/src)

spec = importlib.util.spec_from_file_location("iu", REPO + "/src/inference_utils.py")
iu = importlib.util.module_from_spec(spec); spec.loader.exec_module(iu)

NAMES = ("tf_copy_cos", "tf_best_unfed_cos", "tf_copy_match")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batches", type=int, default=40)
    a = ap.parse_args()
    model, tokenizer, _ = iu.load_retriever(a.ckpt)
    opt = model._train_opt
    model = model.cuda().eval()
    torch.manual_seed(0)
    train_loader, _ = finetuning_multi.prepare_data(opt, tokenizer)
    sums = {n: [0.0, 0.0] for n in NAMES}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b, batch in enumerate(train_loader):
            if b == a.batches:
                break
            stats = {}
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            _, stats = model(**batch, stats_prefix="train", sampling_rate=0.0)
            for n in NAMES:  # absent when the batch's questions have one gold (k = 1)
                if f"train/{n}" in stats:
                    v, c = stats[f"train/{n}"]
                    sums[n][0] += v * c; sums[n][1] += c
    if not any(c for _, c in sums.values()):
        print("no copy stats; keys seen:", sorted(stats), flush=True)
    out = {n: round(s / c, 3) if c else None for n, (s, c) in sums.items()}
    out["n_outputs"] = int(sums["tf_copy_cos"][1])
    out.update(ckpt=a.ckpt, loss_fn=getattr(opt, "loss_fn", None), batches=a.batches)
    print("RESULT", json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
