"""finetuning_multi_paper.py with the retriever as a parameter (session N, 2026-11-12).

  --retriever inf        (default) exactly finetuning_multi_paper.py: infly/inf-retriever-v1-1.5b
  --retriever nemotron   nvidia/Nemotron-3.5-Embed-1B-BF16 (src/nemotron.py) as query model and frozen doc encoder:
                         mean pooling, 'query: ' / 'passage: ' inputs without BOS / EOS, bidirectional. Frozen doc
                         encoder only (--freeze_doc_encoder). Needs the nemotron.ext3 overlay (transformers 5).
                         With --force_causal: causal query encoder + last-token pooling (docs unchanged).
  --retriever qwen3      Qwen/Qwen3-Embedding-0.6B (src/qwen3emb.py): causal, last-token pooling, as the inf path;
                         queries 'Instruct: <task>\nQuery:<q>'. Frozen doc encoder only. div.ext3 (transformers 4.56).

The choice is saved in the checkpoint's opt (opt.retriever, opt.retriever_model_id); src/inference_utils.load_retriever
reads it at retrieval time.

  --one_label_shuffled   (with --one_label_step; 2026-11-16, user: match the legacy --mix_one_label_shuffled) the
                         second update of every step uses OTHER questions, as the legacy loop's separate shuffled
                         one-label loader did, instead of the current batch's questions:
                           multi:  one-gold batch (per question one random gold + one of its negatives, k = 1) built
                                   from per_gpu_batch_size questions drawn at random from the whole training set;
                           single: an ordinary single-query batch of such random questions (instead of a redraw of the
                                   same questions, finetuning_multi_paper._install_single_redraw).
                         The extra questions are drawn uniformly with replacement by a per-rank RNG (legacy: without
                         replacement within an epoch; at ~45 epochs of draws the difference is negligible). Default
                         off: without it the script behaves exactly as before.
"""
import logging
import random

import torch

import finetuning_multi
import finetuning_multi_paper
from src import inbatch

logger = logging.getLogger(__name__)

_ORIG_PREPARE_DATA = finetuning_multi.prepare_data  # before PaperOptions' single-mode redraw wraps it


class _ShuffledPairDataset(torch.utils.data.Dataset):
    """Item i = (base[i], base[j]), j uniform over the training set from this dataset's own RNG."""

    def __init__(self, base, seed):
        self.base = base
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return self.base[i], self.base[self.rng.randrange(len(self.base))]

    def __getattr__(self, name):  # gold_counts etc.
        if name in ("base", "rng"):
            raise AttributeError(name)
        return getattr(self.base, name)


def _install_one_label_shuffled():
    from torch.utils.data import DataLoader, RandomSampler
    from src import dist_utils, finetuning_data

    def prepare_data_shuffled(opt, tokenizer):
        train_loader, eval_loader = _ORIG_PREPARE_DATA(opt, tokenizer)
        collate = train_loader.collate_fn
        multi = opt.training_mode == "multi"
        logged = [0]

        def one_gold(ex):
            return dict(ex, gold=[random.choice(ex["gold"])], negatives=[random.choice(ex["negatives"])])

        def collate_pair(items):
            batch = collate([a for a, _ in items])
            others = [b for _, b in items]
            batch["_redraw"] = collate([one_gold(b) for b in others] if multi else others)
            if logged[0] < 3:  # smoke check: the extra batch should share almost no questions with the main one
                logged[0] += 1
                qa = {a["query"] for a, _ in items}
                same = sum(b["query"] in qa for b in others)
                logger.warning(f"[one_label_shuffled] batch of {len(items)}: {same} extra questions also in the main "
                               f"batch; main k={len(items[0][0]['gold']) if multi else 1}; "
                               f"extra k per question={[len(one_gold(b)['gold']) for b in others[:3]] if multi else 1}")
            return batch

        ds = _ShuffledPairDataset(train_loader.dataset, 1000003 * opt.seed + 7919 * (dist_utils.get_rank() + 1))
        if isinstance(train_loader.batch_sampler, finetuning_data.GoldLengthGroupedBatchSampler):
            loader = DataLoader(ds, batch_sampler=train_loader.batch_sampler, num_workers=0, pin_memory=False,
                                collate_fn=collate_pair)
        else:
            loader = DataLoader(ds, sampler=RandomSampler(ds), batch_size=train_loader.batch_size, drop_last=True,
                                num_workers=0, pin_memory=False, collate_fn=collate_pair)
        return loader, eval_loader

    finetuning_multi.prepare_data = prepare_data_shuffled


class NemoOptions(finetuning_multi_paper.PaperOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument("--retriever", type=str, default="inf", choices=["inf", "nemotron", "qwen3"],
                                 help="retriever backbone (query model and frozen doc encoder); default inf-retriever")
        self.parser.add_argument("--one_label_shuffled", action="store_true",
                                 help="with --one_label_step: the extra update uses other (random) questions, as legacy")

    def parse(self):
        opt = super().parse()  # PaperOptions applies its class / data substitutions first
        if opt.one_label_shuffled:
            if not opt.one_label_step:
                self.parser.error("--one_label_shuffled needs --one_label_step")
            if opt.negative_ctxs > 1:
                self.parser.error("--one_label_shuffled is implemented for --negative_ctxs 1 only")
            _install_one_label_shuffled()  # replaces the single-mode redraw too
            logger.warning("[one_label_shuffled] extra update on random other questions (legacy mix_one_label_shuffled)")
        if opt.retriever == "nemotron":
            if not opt.freeze_doc_encoder:
                self.parser.error("--retriever nemotron is implemented for --freeze_doc_encoder only")
            # --force_causal (2026-11-14): causal query encoder + last-token pooling; the frozen doc encoder stays
            # bidirectional + mean pooling (src/nemotron.NemotronMixin).
            # --io_proj (2026-11-13): ProjOptions (in super().parse()) has already bound inbatch's frozen classes to
            # the inbatch_proj classes, so the Nemotron wrap below sits on top of them (load_retriever does the same).
            from src import nemotron
            for name in ("EmbeddingModelFrozenDocEnc", "EmbeddingModelFrozenDocEncSingleQuery"):
                setattr(inbatch, name, nemotron.nemotron_class(getattr(inbatch, name)))
            nemotron.use_nemotron_text_format()
            logger.warning(f"[retriever] {nemotron.NEMOTRON_ID}")
        elif opt.retriever == "qwen3":
            if not opt.freeze_doc_encoder:
                self.parser.error("--retriever qwen3 is implemented for --freeze_doc_encoder only")
            if opt.force_causal:
                self.parser.error("--retriever qwen3 is causal already; drop --force_causal")
            from src import qwen3emb
            for name in ("EmbeddingModelFrozenDocEnc", "EmbeddingModelFrozenDocEncSingleQuery"):
                setattr(inbatch, name, qwen3emb.qwen3_class(getattr(inbatch, name)))
            qwen3emb.use_qwen3_text_format()
            logger.warning(f"[retriever] {qwen3emb.QWEN3_ID}")
        return opt


if __name__ == "__main__":
    finetuning_multi.Options = NemoOptions
    finetuning_multi_paper._install_dev_loss_eval()
    finetuning_multi.main()
