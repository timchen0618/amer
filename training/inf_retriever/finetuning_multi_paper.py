"""finetuning_multi_proj.py plus the paper's (legacy stack) training choices, for Session F (2026-10-22).

Runs finetuning_multi.main() unchanged, after in-process substitutions; finetuning_multi.py, options.py and
inbatch.py are not edited. Every option defaults to off, so without them this is finetuning_multi_proj.py.

  --sampling_cap c        clip the scheduled-sampling rate at c (0 < c <= 1). With --sampling_saturate_frac 0.1
                          and --sampling_cap 0.8 the rate is min(10 * step / total_steps, 0.8), the legacy
                          train_distributed.py schedule (sample_rate_multiplier 10, cap 0.8). The paper's
                          "multi_sampling" variant is the existing --full_sampling (rate 1.0 throughout).
  --one_label_step        the legacy --mix_one_label_shuffled step: after every optimizer step, a second forward /
                          backward / optimizer step on a one-gold version of the same batch (each question keeps one
                          random gold and one of its negatives, so k = 1: the first output alone, contrastive
                          against in-batch documents). The LR scheduler still steps once per training step, as in
                          the legacy loop. Difference from legacy: legacy drew the one-label batch from a separate
                          shuffled loader (other questions); here it reuses the current batch's questions.
                          Single mode (--training_mode standard_org_q, k = 1 already): the extra step uses a second,
                          independent draw of the same questions (new random gold and negatives), see
                          _install_single_redraw.
  --negative_ctxs n > 1   n negatives per gold (default 1) for frozen multi-query runs with --loss_fn hungarian;
                          the shared collator / document split / loss assume one, see _enable_multi_negatives.
  --io_proj_init random   (with --io_proj) nn.Linear's default init for the input/output layers, as in legacy
                          src/model.py, instead of identity (src/inbatch_proj.py).
  --query_instruct_task t (or "paper") the query instruction ("Instruct: t\nQuery: q") in training and in-training eval; saved in
                          the checkpoint's opt, which retrieval_inf.py then uses (2026-10-27, P3: the paper's prompt).
  --save_best_dev_loss    keep checkpoint/best_dev_loss/: the lowest eval_loss_sr1 so far (legacy save_only_improve)

    tools/phaseC_train.sbatch with TRAIN_SCRIPT=finetuning_multi_paper.py and EXTRA="... --sampling_cap 0.8 ..."
"""
import logging
import torch

import finetuning_multi
import finetuning_multi_proj
from src import inbatch

logger = logging.getLogger(__name__)
_STATE = {"batch": None, "batch2": None, "n": 0, "loss_sum": 0.0}


class PaperOptions(finetuning_multi_proj.ProjOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument("--sampling_cap", type=float, default=1.0,
                                 help="clip the scheduled-sampling rate at this value (legacy: 0.8)")
        self.parser.add_argument("--one_label_step", action="store_true",
                                 help="legacy --mix_one_label_shuffled: an extra k=1 step per training step")
        self.parser.add_argument("--io_proj_init", type=str, default="identity", choices=["identity", "random"],
                                 help="init of the --io_proj layers: identity (default) or nn.Linear default (legacy)")
        self.parser.add_argument("--query_instruct_task", type=str, default=None,
                                 help="query instruction for training and retrieval (default: finetuning_data's)")
        self.parser.add_argument("--save_best_dev_loss", action="store_true",
                                 help="save checkpoint/best_dev_loss whenever eval_loss_sr1 reaches a new minimum")

    def parse(self):
        opt = super().parse()  # ProjOptions binds the --io_proj class
        if not 0.0 < opt.sampling_cap <= 1.0:
            self.parser.error(f"--sampling_cap must be in (0, 1], got {opt.sampling_cap}")
        if opt.io_proj_init != "identity" and not opt.io_proj:
            self.parser.error("--io_proj_init needs --io_proj")
        if opt.training_mode == "multi" and opt.negative_ctxs > 1:
            if not opt.freeze_doc_encoder:
                self.parser.error("--negative_ctxs > 1 is implemented for the frozen doc encoder only")
            _enable_multi_negatives(opt.negative_ctxs)
        if opt.sampling_cap < 1.0:
            _cap_sampling(opt.sampling_cap)
        if opt.one_label_step:
            _install_one_label_step()
            if opt.training_mode == "standard_org_q":
                _install_single_redraw()
        if opt.query_instruct_task == "paper":  # shorthand: sbatch EXTRA strings cannot carry spaces
            opt.query_instruct_task = PAPER_TASK
        if opt.query_instruct_task:
            _set_query_task(opt.query_instruct_task)
        return opt


# The legacy stack's instruction (create_input_for_contrievers.py / gen_ret_and_eval.py, INF base model).
PAPER_TASK = "Retrieve a diverse set of documents that covers multiple aspect of the query."


def _set_query_task(task):
    """finetuning_data builds every query as get_detailed_instruct(<its fixed task>, q); swap in `task`."""
    from src import finetuning_data

    def get_detailed_instruct(task_description, query):
        return f"Instruct: {task}\nQuery: {query}"

    finetuning_data.get_detailed_instruct = get_detailed_instruct
    logger.warning(f"[query_instruct_task] {task!r}")


def _cap_sampling(cap):
    """Rebind the frozen training class to a subclass whose forward clips sampling_rate at cap.

    Only EmbeddingModelFrozenDocEnc: its __init__ uses zero-argument super(), so rebinding its module name is safe
    (as --io_proj already does). EmbeddingModelDocEncNoProj.__init__ calls super(EmbeddingModelDocEncNoProj, self),
    a by-name lookup, so rebinding that name breaks construction (TypeError in super())."""
    for name in ("EmbeddingModelFrozenDocEnc",):
        base = getattr(inbatch, name)

        def forward(self, *args, sampling_rate=1.0, __base=base, **kwargs):
            eff = min(sampling_rate, cap)
            if self.training:
                _STATE["cap_calls"] = _STATE.get("cap_calls", 0) + 1
                if _STATE["cap_calls"] in (1, 25) or _STATE["cap_calls"] % 250 == 0:
                    logger.info(f"[sampling_cap] call {_STATE['cap_calls']}: scheduled {sampling_rate:.3f} -> used {eff:.3f}")
            return __base.forward(self, *args, sampling_rate=eff, **kwargs)

        setattr(inbatch, name, type(base.__name__, (base,), {"forward": forward, "__module__": base.__module__}))


def _enable_multi_negatives(n):
    """--negative_ctxs n > 1 for multi-query training (2026-10-22, Stage C): n negatives per gold instead of one.

    The shared code assumes one negative per gold in three places, each overridden here without editing it:
      1. CollatorDocEncMultiQuery asserts len(negatives) == len(golds): replaced by a subclass that tokenizes
         golds + all negatives (still golds first, then negatives in example order);
      2. encode_documents splits the passage embeddings in half: the frozen class's split is now 1 : n;
      3. HungarianContrastiveLoss views the negatives as (batch * k, d): its subclass reshapes them to (-1, d).
    The dataset already draws negative_ctxs * k negatives per question (finetuning_data.Dataset)."""
    from src import finetuning_data

    class CollatorMultiNeg(finetuning_data.CollatorDocEncMultiQuery):
        def __call__(self, batch):
            import random
            for ex in batch:  # the in-training eval set is built with one negative per gold: resample to n per gold
                need = n * len(ex["gold"]) - len(ex["negatives"])
                if need > 0:
                    ex["negatives"] = list(ex["negatives"]) + random.choices(ex["negatives"], k=need)
            kept = [ex["negatives"] for ex in batch]
            for ex in batch:  # parent builds the query side with one negative per gold
                ex["negatives"] = ex["negatives"][:len(ex["gold"])]
            out = super().__call__(batch)
            for ex, negs in zip(batch, kept):
                ex["negatives"] = negs
            golds = [g for ex in batch for g in ex["gold"]]
            negs = [x for ex in batch for x in ex["negatives"]]
            assert len(negs) == n * len(golds), (len(negs), n, len(golds))
            kout = self.tokenizer.batch_encode_plus(golds + negs, max_length=self.passage_maxlength, truncation=True,
                                                    padding=True, add_special_tokens=True, return_tensors="pt")
            kt, km = kout["input_ids"], kout["attention_mask"].to(torch.long)
            out.update(k_tokens=kt, k_mask=km, g_tokens=kt[:len(golds)], g_mask=km[:len(golds)],
                       n_tokens=kt[len(golds):], n_mask=km[len(golds):])
            return out

    finetuning_data.CollatorDocEncMultiQuery = CollatorMultiNeg

    class HungarianMultiNeg(inbatch.HungarianContrastiveLoss):
        def _core_forward(self, outputs, positive_embeddings, negative_embeddings):
            b, k, d = outputs.shape
            # the parent views negatives as (b * k, d); pass them through a (b, k, d)-shaped view of the pool instead
            return _hungarian_any_negatives(self, outputs, positive_embeddings, negative_embeddings.reshape(-1, d))

    base = inbatch.EmbeddingModelFrozenDocEnc

    class MultiNeg(base):
        def __init__(self, opt, *args, **kwargs):
            super().__init__(opt, *args, **kwargs)
            assert type(self.loss_fct) is inbatch.HungarianContrastiveLoss, \
                f"--negative_ctxs > 1 is implemented for --loss_fn hungarian only, got {type(self.loss_fct).__name__}"
            new = HungarianMultiNeg.__new__(HungarianMultiNeg)
            new.__dict__.update(self.loss_fct.__dict__)
            self.loss_fct = new

        def encode_documents(self, input_document_ids, attention_mask_document):
            n_pos = input_document_ids.size(0) // (1 + n)
            assert n_pos * (1 + n) == input_document_ids.size(0)
            chunks = []
            with torch.no_grad():  # frozen encoder, no gradient: encode in chunks so many negatives fit in memory
                for i in range(0, input_document_ids.size(0), 1024):
                    ids, mask = input_document_ids[i:i + 1024], attention_mask_document[i:i + 1024]
                    out = self.doc_encoder(input_ids=ids, attention_mask=mask)
                    chunks.append(self.last_token_pool(out.last_hidden_state, mask))
            emb = torch.cat(chunks)
            return emb[:n_pos], emb[n_pos:]

    MultiNeg.__name__ = base.__name__
    inbatch.EmbeddingModelFrozenDocEnc = MultiNeg


def _hungarian_any_negatives(self, outputs, positive_embeddings, negative_flat):
    """inbatch.HungarianContrastiveLoss._core_forward with the negatives pool of any size (copied; one line differs)."""
    import torch.nn.functional as F
    from src import dist_utils
    if self.normalize_embeddings:
        outputs = F.normalize(outputs, dim=-1)
        positive_embeddings = F.normalize(positive_embeddings, dim=-1)
        negative_flat = F.normalize(negative_flat, dim=-1)
    batch_size, k, d = outputs.shape
    outputs = outputs.view(batch_size * k, d)
    positive_embeddings = positive_embeddings.reshape(batch_size * k, d)
    all_embeddings = torch.cat([positive_embeddings, negative_flat], dim=0)  # was negative.view(batch_size * k, d)
    var_sizes = dist_utils.get_varsize(all_embeddings)
    gather_kemb = dist_utils.varsize_gather(all_embeddings)
    similarity = self.log_softmax(torch.einsum('bd,cd->bc', outputs / self.temperature, gather_kemb))
    losses = []
    assigned_gold = torch.empty(batch_size, k, dtype=torch.long)
    local_start = 0 if dist_utils.get_rank() == 0 else int(var_sizes[:dist_utils.get_rank()].sum())
    for i in range(batch_size):
        start_idx = local_start + k * i
        cost_matrix = similarity[k * i:k * (i + 1)][:, start_idx:start_idx + k]
        row_ind, col_ind = self.linear_sum_assignment(cost_matrix.float().detach().cpu().numpy(), maximize=True)
        assigned_gold[i, torch.as_tensor(row_ind)] = torch.as_tensor(col_ind, dtype=torch.long)
        losses.append(-(cost_matrix[row_ind, col_ind]).mean())
    self.last_assignment = assigned_gold.to(outputs.device)
    self.last_stats = inbatch._hungarian_accuracy(similarity.detach(), assigned_gold.to(outputs.device), local_start, k)
    return torch.stack(losses).mean()


def one_label_batch(batch):
    """k = 1 version of a CollatorDocEncMultiQuery batch: per question one random gold and one of its negatives."""
    q = batch["q_tokens"]
    bsz, n = q.size(0), batch["k_tokens"].size(0)
    k = n // (2 * bsz)
    dev = batch["k_tokens"].device
    j = torch.randint(k, (bsz,), device=dev)
    pos = torch.arange(bsz, device=dev) * k + j                   # gold rows (first half)
    neg = bsz * k + torch.arange(bsz, device=dev) * k + torch.randint(k, (bsz,), device=dev)
    rows = torch.cat([pos, neg])
    out = {key: batch[key] for key in ("q_tokens", "q_mask", "q_position_ids")}
    out["k_tokens"], out["k_mask"] = batch["k_tokens"][rows], batch["k_mask"][rows]
    return out


class _BatchRecorder:
    """Wraps the training DataLoader so the optimizer hook can see the current batch (same dict object)."""

    def __init__(self, loader):
        self._loader = loader

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        for b in self._loader:
            _STATE["batch2"] = b.pop("_redraw", None)  # single mode: the second draw (see _install_single_redraw)
            _STATE["batch"] = b
            yield b

    def __getattr__(self, name):
        return getattr(self._loader, name)


def _install_one_label_step():
    build = finetuning_multi.build_accelerator

    def build_accelerator(*args, **kwargs):
        acc = build(*args, **kwargs)
        prepare = acc.prepare

        def prepare_and_hook(*objs):
            model, optimizer, loader, scheduler = prepare(*objs)
            step = optimizer.step

            def step_with_one_label(*a, **kw):
                out = step(*a, **kw)
                optimizer.zero_grad()
                if _STATE["batch2"] is not None:  # single mode: same questions, independently redrawn gold / negatives
                    extra = {key: v.to(acc.device) if isinstance(v, torch.Tensor) else v
                             for key, v in _STATE["batch2"].items()}
                else:
                    extra = one_label_batch(_STATE["batch"])
                loss, _ = model(**extra, stats_prefix="train/one_label")
                acc.backward(loss)
                step(*a, **kw)
                _STATE["n"] += 1
                _STATE["loss_sum"] += loss.item()
                if _STATE["n"] % 25 == 0:
                    logger.info(f"[one_label_step] steps={_STATE['n']} mean_loss_last25="
                                f"{_STATE['loss_sum'] / 25:.4f}")
                    _STATE["loss_sum"] = 0.0
                return out

            optimizer.step = step_with_one_label
            return model, optimizer, _BatchRecorder(loader), scheduler

        acc.prepare = prepare_and_hook
        return acc

    finetuning_multi.build_accelerator = build_accelerator


class _RedrawDataset(torch.utils.data.Dataset):
    """Each item = two independent draws of the same training question (Dataset.__getitem__ picks a random gold and
    random negatives on every call)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.base[i], self.base[i]


def _install_single_redraw():
    """--one_label_step in single mode (2026-11-03, user: step-matched single baseline with two updates per step).

    A single-query batch already has k = 1, so one_label_batch would return the same batch and the extra update would
    repeat the first. Instead the train loader yields, for the same questions, a second independently drawn batch
    (new random gold and negatives); the extra update uses it. Sampler, batch size and drop_last are unchanged."""
    from torch.utils.data import DataLoader, RandomSampler

    prepare_data = finetuning_multi.prepare_data

    def prepare_data_redraw(opt, tokenizer):
        train_loader, eval_loader = prepare_data(opt, tokenizer)
        collate = train_loader.collate_fn

        def collate_redraw(items):
            batch = collate([a for a, _ in items])
            batch["_redraw"] = collate([b for _, b in items])
            if _STATE.get("redraw_logged", 0) < 3:  # smoke check: the two draws should differ for most questions
                _STATE["redraw_logged"] = _STATE.get("redraw_logged", 0) + 1
                same = sum(a["gold"] == b["gold"] for a, b in items)
                logger.warning(f"[single_redraw] batch of {len(items)}: same gold in both draws for {same}; "
                               f"q0 gold1={items[0][0]['gold'][:60]!r} gold2={items[0][1]['gold'][:60]!r}")
            return batch

        ds = _RedrawDataset(train_loader.dataset)
        loader = DataLoader(ds, sampler=RandomSampler(ds), batch_size=train_loader.batch_size, drop_last=True,
                            num_workers=0, pin_memory=False, collate_fn=collate_redraw)
        return loader, eval_loader

    finetuning_multi.prepare_data = prepare_data_redraw


def _install_dev_loss_eval():
    """Log the legacy selection metric at every in-training eval (2026-10-23, user request: compare selection rules).

    The legacy loop (train_distributed.py) kept the checkpoint with the lowest dev Hungarian loss computed at
    sampling rate 1.0. finetuning_multi.evaluate() logs in-batch retrieval metrics only; this wraps it to also
    compute the training loss (opt.loss_fn) over the same eval batches with sampling_rate=1.0, using the model
    evaluate() just built, and logs it as eval_loss_sr1 (wandb) and as a "[eval_loss_sr1] step S: v" line that
    tools/f_run.sh reads. evaluate() runs on the main process only, so the loss's cross-rank gather is replaced by
    the identity while it runs."""
    from src import dist_utils, utils
    orig_eval = finetuning_multi.evaluate
    # The eval model is recorded where evaluate() loads its weights (2026-10-27; was a subclass of
    # EmbeddingModelFrozenDocEnc, which missed the --io_proj classes and single-query runs).
    load_state = inbatch.load_model_state

    def load_model_state(model, state_dict, *a, **k):
        _STATE["last_model"] = model
        return load_state(model, state_dict, *a, **k)

    inbatch.load_model_state = load_model_state

    class _NoScheduler:
        def state_dict(self):
            return None

    def evaluate(opt, state_dict, eval_loader, accelerator, step, device):
        out = orig_eval(opt, state_dict, eval_loader, accelerator, step, device)
        model = _STATE.pop("last_model", None)
        if model is None:
            return out
        saved = (dist_utils.varsize_gather, dist_utils.get_varsize, dist_utils.get_rank)
        dist_utils.varsize_gather = lambda x, *a, **k: x
        dist_utils.get_varsize = lambda x, *a, **k: torch.tensor([x.size(0)], device=x.device)
        dist_utils.get_rank = lambda: 0
        tot, n = 0.0, 0
        try:  # never let the extra metric break a training run
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for batch in eval_loader:
                    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
                    loss, _ = model(**batch, stats_prefix="eval_sr1", sampling_rate=1.0)
                    bsz = batch["q_tokens"].size(0)
                    tot += loss.item() * bsz
                    n += bsz
        except Exception as e:
            logger.warning(f"[eval_loss_sr1] step {step}: FAILED ({type(e).__name__}: {e})")
            n = 0
        finally:
            dist_utils.varsize_gather, dist_utils.get_varsize, dist_utils.get_rank = saved
        if n:
            logger.warning(f"[eval_loss_sr1] step {step}: {tot / n:.5f}")
            accelerator.log({"eval_loss_sr1": tot / n}, step=step)
            if getattr(opt, "save_best_dev_loss", False) and tot / n < _STATE.get("best_dev_loss", float("inf")):
                _STATE["best_dev_loss"] = tot / n
                utils.save_state_dict(state_dict, None, _NoScheduler(), step, opt,
                                      opt.output_dir + opt.run_name, "best_dev_loss")
                logger.warning(f"[best_dev_loss] step {step}: {tot / n:.5f} saved")
        del model
        torch.cuda.empty_cache()
        return out

    finetuning_multi.evaluate = evaluate


if __name__ == "__main__":
    finetuning_multi.Options = PaperOptions
    _install_dev_loss_eval()
    finetuning_multi.main()
