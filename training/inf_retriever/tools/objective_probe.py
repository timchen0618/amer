"""Objective-audit probes D2 (copying) and D3 (gradient into output 0) on saved checkpoints.

reports/objective_audit_2026-10-13/PLAN.md. One process, one GPU, no training, dev data.

For each checkpoint (or "base" = the HF weights in the class a reference checkpoint's opt selects):

D2, no gradients, model.forward on dev batches at sampling rate 0 (every input teacher-forced) and 1
(every input the model's own output); reads the copy statistics the forward logs (inbatch.copy_stats:
tf_copy_cos, tf_best_unfed_cos, tf_copy_match, free_repeat_cos, free_repeat_top1) and adds
  tf_top1_fed     share of teacher-forced outputs t >= 1 whose top-1 over the in-batch pool is gold t-1
  tf_top1_unfed   ... is a gold not fed in before t
  free_top1_gold  share of free-running outputs whose top-1 is any gold of the question
  norm_gold / norm_token / norm_output   mean L2 norm of a fed gold (raw doc embedding), of the
                  question's token embeddings, and of a raw output embedding (what --fed_norm compares).

D3, at sampling rates 0.5 and 1: the gradient of the run's own training loss with respect to output 0
(q0, the question's embedding), split into g_own (q0's own loss term) and g_feed (later outputs'
terms, reaching q0 through the fed-back input). Computed as g_own = dL/dq0 with the fed-back outputs
detached (same draw, same forward values, same assignment) and g_feed = dL/dq0 - g_own. Logs per-example
means of |g_own|, |g_feed|, |g_feed| / |g_own| and cos(g_own, g_feed), and the same for the batch-summed
vectors.

  python training/inf_retriever/tools/objective_probe.py --out <json> --ckpts base:<ref_ckpt_dir> <ckpt_dir> ...
ckpt_dir = checkpoints/qampari/<run>/checkpoint/step-N (contains checkpoint.pth).
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from src import finetuning_data, inbatch  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(spec, device):
    """spec = <ckpt_dir> or base:<ref_ckpt_dir>. Returns (model, opt, label)."""
    base = spec.startswith("base:")
    ckpt_dir = spec[5:] if base else spec
    sd = torch.load(os.path.join(ckpt_dir, "checkpoint.pth"), map_location="cpu", weights_only=False)
    opt = sd["opt"]
    cls = inbatch.EmbeddingModelFrozenDocEnc if getattr(opt, "freeze_doc_encoder", False) else inbatch.EmbeddingModelDocEncNoProj
    model = cls(opt, None, None)
    if not base:
        inbatch.load_model_state(model, sd["model"])
    del sd
    model = model.to(device)
    model.train()   # Qwen2 has no dropout; train mode keeps gradient checkpointing on for D3
    label = ("base(" + cls.__name__ + ")") if base else ckpt_dir
    return model, opt, label


class Capture(torch.nn.Module):
    """Wraps the run's loss and keeps its inputs."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.uses_feed_info = getattr(inner, "uses_feed_info", False)
        self.seen = None

    @property
    def last_assignment(self):
        return getattr(self.inner, "last_assignment", None)

    def forward(self, outputs, pos, neg, **kw):
        self.seen = dict(outputs=outputs, pos=pos, neg=neg, fed_gold=kw.get("fed_gold"))
        return self.inner(outputs, pos, neg, **kw)


def loader(opt, tokenizer, eval_data, batch_size):
    ds = finetuning_data.Dataset(datapaths=[eval_data], normalize=False, training_mode="multi",
                                 global_rank=-1, world_size=-1, maxload=None, training=False)
    sampler = finetuning_data.GoldLengthGroupedBatchSampler(ds.gold_counts, batch_size, drop_last=False, shuffle=False)
    coll = finetuning_data.CollatorDocEncMultiQuery(tokenizer, passage_maxlength=opt.chunk_length)
    return DataLoader(ds, batch_sampler=sampler, num_workers=0, collate_fn=coll)


def fwd(model, batch, rate, seed):
    torch.manual_seed(seed)
    k_tokens = torch.cat([batch["g_tokens"], batch["n_tokens"]], 0)
    k_mask = torch.cat([batch["g_mask"], batch["n_mask"]], 0)
    return model(q_tokens=batch["q_tokens"], q_mask=batch["q_mask"], q_position_ids=batch["q_position_ids"],
                 k_tokens=k_tokens, k_mask=k_mask, stats_prefix="p", sampling_rate=rate)


class Acc:
    def __init__(self):
        self.s, self.n = {}, {}

    def add(self, name, total, count):
        self.s[name] = self.s.get(name, 0.0) + float(total)
        self.n[name] = self.n.get(name, 0) + int(count)

    def means(self):
        return {k: self.s[k] / self.n[k] for k in sorted(self.s) if self.n[k]}


@torch.no_grad()
def d2(model, batches, device):
    out = {}
    for rate in (0.0, 1.0):
        acc = Acc()
        for bi, b in enumerate(batches):
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            _, st = fwd(model, b, rate, seed=bi)
            for name in ("tf_copy_cos", "tf_best_unfed_cos", "tf_copy_match", "free_repeat_cos", "free_repeat_top1"):
                if f"p/{name}" in st:
                    v, c = st[f"p/{name}"]
                    acc.add(name, v * c, c)
            seen = model.loss_fct.seen
            q = F.normalize(seen["outputs"].float(), dim=-1)                    # (b, k, d)
            bsz, k, d = q.shape
            golds = F.normalize(seen["pos"].float().reshape(bsz * k, d), dim=-1)
            pool = torch.cat([golds, F.normalize(seen["neg"].float().reshape(-1, d), dim=-1)])
            top1 = torch.einsum("bkd,nd->bkn", q, pool).argmax(-1)              # (b, k)
            gidx = top1 - k * torch.arange(bsz, device=device).unsqueeze(1)    # gold index if in own block
            own = (top1 >= k * torch.arange(bsz, device=device).unsqueeze(1)) & (gidx < k) & (gidx >= 0)
            if rate == 0.0:   # every gold j < t was fed in before output t
                t = torch.arange(1, k, device=device)
                is_fed = own[:, 1:] & (gidx[:, 1:] == (t - 1))
                unfed_hit = own[:, 1:] & (gidx[:, 1:] >= t)
                acc.add("tf_top1_fed", is_fed.sum(), bsz * (k - 1))
                acc.add("tf_top1_unfed", unfed_hit.sum(), bsz * (k - 1))
            else:
                acc.add("free_top1_gold", own[:, 1:].sum(), bsz * (k - 1))
                for t in range(1, min(k, 5)):   # per position: is output t drifting toward output 0? (D5, hps)
                    acc.add(f"free_cos_q{t}_q0", (q[:, t] * q[:, 0]).sum(), bsz)
                    acc.add(f"free_cos_q{t}_prev", (q[:, t] * q[:, t - 1]).sum(), bsz)
            m = b["q_mask"].unsqueeze(-1).float()
            tok = model.embedding(b["q_tokens"]).float().norm(dim=-1, keepdim=True)
            acc.add("norm_token", (tok * m).sum(), m.sum())
            acc.add("norm_gold", seen["pos"].float().norm(dim=-1).sum(), seen["pos"].reshape(-1, d).size(0))
            acc.add("norm_output", seen["outputs"].float().norm(dim=-1).sum(), bsz * k)
        out[f"rate_{rate:g}"] = acc.means()
    return out


def d3(model, batches, device):
    out = {}
    orig_pool = model.last_token_pool
    for rate in (0.5, 1.0):
        acc = Acc()
        for bi, b in enumerate(batches):
            b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in b.items()}
            grads = []
            for detach in (False, True):
                calls = []

                def pool(h, m):
                    r = orig_pool(h, m)
                    calls.append(r)
                    return r
                model.last_token_pool = pool
                model.detach_fed_output = detach
                loss, _ = fwd(model, b, rate, seed=1000 + bi)
                bsz = b["q_tokens"].size(0)
                q0 = calls[1]                       # call 0 = documents, call 1 = output 0
                assert calls[0].size(0) != bsz and q0.size(0) == bsz, [c.shape for c in calls]
                g, = torch.autograd.grad(loss, q0)
                grads.append(g.float())
                del loss, calls
            model.last_token_pool = orig_pool
            model.detach_fed_output = False
            g_all, g_own = grads
            g_feed = g_all - g_own
            n_own, n_feed = g_own.norm(dim=-1), g_feed.norm(dim=-1)
            valid = n_own > 0
            acc.add("own_norm", n_own.sum(), bsz)
            acc.add("feed_norm", n_feed.sum(), bsz)
            acc.add("feed_over_own", (n_feed[valid] / n_own[valid]).sum(), int(valid.sum()))
            cos = F.cosine_similarity(g_own, g_feed, dim=-1)
            ok = valid & (n_feed > 0)
            acc.add("cos_own_feed", cos[ok].sum(), int(ok.sum()))
            acc.add("share_feed_opposes", (cos[ok] < 0).sum(), int(ok.sum()))
            acc.add("batch_cos_own_feed", F.cosine_similarity(g_own.sum(0), g_feed.sum(0), dim=0), 1)
            acc.add("batch_feed_over_own", g_feed.sum(0).norm() / g_own.sum(0).norm().clamp_min(1e-12), 1)
        out[f"rate_{rate:g}"] = acc.means()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--eval_data", default="/scratch/hc3337/projects/autoregressive/data/training/clean_v3/qampari/dev_data.jsonl")
    ap.add_argument("--d2_batches", type=int, default=10)
    ap.add_argument("--d3_batches", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--skip_d3", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    device = torch.device("cuda")
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for spec in a.ckpts:
        if spec in results:
            log(f"skip {spec} (done)"); continue
        t0 = time.time()
        model, opt, label = load(spec, device)
        model.loss_fct = Capture(model.loss_fct)
        dl = loader(opt, model.tokenizer, a.eval_data, a.batch_size)
        it = iter(dl)
        batches = [next(it) for _ in range(max(a.d2_batches, a.d3_batches))]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            r = {"label": label, "loss_fn": getattr(opt, "loss_fn", None),
                 "freeze_doc_encoder": getattr(opt, "freeze_doc_encoder", False),
                 "d2": d2(model, batches[:a.d2_batches], device)}
            if not a.skip_d3:
                r["d3"] = d3(model, batches[:a.d3_batches], device)
        results[spec] = r
        json.dump(results, open(a.out, "w"), indent=1)
        log(f"{spec}: {time.time() - t0:.0f}s {json.dumps(r)}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
