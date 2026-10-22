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
  --io_proj_init random   (with --io_proj) nn.Linear's default init for the input/output layers, as in legacy
                          src/model.py, instead of identity (src/inbatch_proj.py).

    tools/phaseC_train.sbatch with TRAIN_SCRIPT=finetuning_multi_paper.py and EXTRA="... --sampling_cap 0.8 ..."
"""
import logging
import torch

import finetuning_multi
import finetuning_multi_proj
from src import inbatch

logger = logging.getLogger(__name__)
_STATE = {"batch": None, "n": 0, "loss_sum": 0.0}


class PaperOptions(finetuning_multi_proj.ProjOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument("--sampling_cap", type=float, default=1.0,
                                 help="clip the scheduled-sampling rate at this value (legacy: 0.8)")
        self.parser.add_argument("--one_label_step", action="store_true",
                                 help="legacy --mix_one_label_shuffled: an extra k=1 step per training step")
        self.parser.add_argument("--io_proj_init", type=str, default="identity", choices=["identity", "random"],
                                 help="init of the --io_proj layers: identity (default) or nn.Linear default (legacy)")

    def parse(self):
        opt = super().parse()  # ProjOptions binds the --io_proj class
        if not 0.0 < opt.sampling_cap <= 1.0:
            self.parser.error(f"--sampling_cap must be in (0, 1], got {opt.sampling_cap}")
        if opt.io_proj_init != "identity" and not opt.io_proj:
            self.parser.error("--io_proj_init needs --io_proj")
        if opt.sampling_cap < 1.0:
            _cap_sampling(opt.sampling_cap)
        if opt.one_label_step:
            _install_one_label_step()
        return opt


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
                loss, _ = model(**one_label_batch(_STATE["batch"]), stats_prefix="train/one_label")
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


if __name__ == "__main__":
    finetuning_multi.Options = PaperOptions
    finetuning_multi.main()
