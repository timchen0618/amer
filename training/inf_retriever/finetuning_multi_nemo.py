"""finetuning_multi_paper.py with the retriever as a parameter (session N, 2026-11-12).

  --retriever inf        (default) exactly finetuning_multi_paper.py: infly/inf-retriever-v1-1.5b
  --retriever nemotron   nvidia/Nemotron-3.5-Embed-1B-BF16 (src/nemotron.py) as query model and frozen doc encoder:
                         mean pooling, 'query: ' / 'passage: ' inputs without BOS / EOS, bidirectional. Frozen doc
                         encoder only (--freeze_doc_encoder). Needs the nemotron.ext3 overlay (transformers 5).
                         With --force_causal: causal query encoder + last-token pooling (docs unchanged).

The choice is saved in the checkpoint's opt (opt.retriever, opt.retriever_model_id); src/inference_utils.load_retriever
reads it at retrieval time.
"""
import logging

import finetuning_multi
import finetuning_multi_paper
from src import inbatch

logger = logging.getLogger(__name__)


class NemoOptions(finetuning_multi_paper.PaperOptions):
    def initialize(self):
        super().initialize()
        self.parser.add_argument("--retriever", type=str, default="inf", choices=["inf", "nemotron"],
                                 help="retriever backbone (query model and frozen doc encoder); default inf-retriever")

    def parse(self):
        opt = super().parse()  # PaperOptions applies its class / data substitutions first
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
        return opt


if __name__ == "__main__":
    finetuning_multi.Options = NemoOptions
    finetuning_multi_paper._install_dev_loss_eval()
    finetuning_multi.main()
