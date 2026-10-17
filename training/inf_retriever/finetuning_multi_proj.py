"""finetuning_multi.py with the learned input/output layers of src/inbatch_proj.py (objective audit, 2026-10-15).

Runs finetuning_multi.main() unchanged, after two in-process substitutions:
  - Options gains --io_proj (saved in the checkpoint's opt, which load_retriever reads to pick the class);
  - with --io_proj, inbatch.EmbeddingModelFrozenDocEnc resolves to EmbeddingModelFrozenDocEncProj, so
    finetuning_multi's model construction and its in-training evaluate() both build the projected model.
Nothing in finetuning_multi.py, options.py or inbatch.py is edited. Frozen document encoder only:
--io_proj requires --freeze_doc_encoder and --training_mode multi.

    tools/phaseC_train.sbatch with TRAIN_SCRIPT=finetuning_multi_proj.py and EXTRA="... --io_proj"
"""
import finetuning_multi
from src import inbatch, inbatch_proj
from src.options import Options


class ProjOptions(Options):
    def initialize(self):
        super().initialize()
        self.parser.add_argument(
            "--io_proj", action="store_true",
            help="learned input layer on fed vectors + output layer on queries (src/inbatch_proj.py)",
        )
        self.parser.add_argument(
            "--io_proj_mode", type=str, default="both", choices=["both", "in", "out"],
            help="with --io_proj: both layers (default), the input layer only, or the output layer only",
        )

    def parse(self):
        opt = super().parse()
        if opt.io_proj_mode != "both" and not opt.io_proj:
            self.parser.error("--io_proj_mode needs --io_proj")
        if opt.io_proj:
            if not (opt.freeze_doc_encoder and opt.training_mode == "multi"):
                self.parser.error("--io_proj needs --freeze_doc_encoder and --training_mode multi")
            inbatch.EmbeddingModelFrozenDocEnc = inbatch_proj.EmbeddingModelFrozenDocEncProj
        return opt


if __name__ == "__main__":
    finetuning_multi.Options = ProjOptions
    finetuning_multi.main()
