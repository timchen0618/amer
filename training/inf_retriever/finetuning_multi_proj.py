"""finetuning_multi.py with the learned input/output layers of src/inbatch_proj.py (objective audit, 2026-10-15).

Runs finetuning_multi.main() unchanged, after two in-process substitutions:
  - Options gains --io_proj (saved in the checkpoint's opt, which load_retriever reads to pick the class);
  - with --io_proj, inbatch.EmbeddingModelFrozenDocEnc resolves to EmbeddingModelFrozenDocEncProj, so
    finetuning_multi's model construction and its in-training evaluate() both build the projected model.
Nothing in finetuning_multi.py, options.py or inbatch.py is edited. --io_proj requires --training_mode multi;
with --freeze_doc_encoder it uses EmbeddingModelFrozenDocEncProj, without it (joint doc encoder, 2026-10-22)
inbatch.EmbeddingModelDocEncNoProj resolves to EmbeddingModelDocEncProj.

    tools/phaseC_train.sbatch with TRAIN_SCRIPT=finetuning_multi_proj.py and EXTRA="... --io_proj"
"""
import finetuning_multi
from src import inbatch, inbatch_proj
from src.options import Options


class _InbatchView:
    """src.inbatch with some attributes overridden, for finetuning_multi's `inbatch.X` lookups only."""

    def __init__(self, **overrides):
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(inbatch, name)


class ProjOptions(Options):
    def initialize(self):
        super().initialize()
        self.parser.add_argument(
            "--io_proj", action="store_true",
            help="learned input layer on fed vectors + output layer on queries (src/inbatch_proj.py)",
        )
        self.parser.add_argument(
            "--io_proj_mode", type=str, default="both", choices=["both", "in", "out", "mlp_in"],
            help="with --io_proj: both layers (default), the input layer only, the output layer only, or a "
                 "2-layer MLP input layer and no output layer (mlp_in; frozen multi only)",
        )

    def parse(self):
        opt = super().parse()
        if opt.io_proj_mode != "both" and not opt.io_proj:
            self.parser.error("--io_proj_mode needs --io_proj")
        if opt.io_proj and opt.training_mode == "standard_org_q":
            # single query (Session F P3-single, 2026-10-27): output layer only, frozen doc encoder only
            if not opt.freeze_doc_encoder or opt.io_proj_mode != "out":
                self.parser.error("single-query --io_proj needs --freeze_doc_encoder and --io_proj_mode out")
            inbatch.EmbeddingModelFrozenDocEncSingleQuery = inbatch_proj.EmbeddingModelFrozenDocEncSingleQueryProj
        elif opt.io_proj:
            if opt.training_mode != "multi":
                self.parser.error("--io_proj needs --training_mode multi or standard_org_q")
            if opt.io_proj_mode == "mlp_in" and not opt.freeze_doc_encoder:
                self.parser.error("--io_proj_mode mlp_in is implemented for --freeze_doc_encoder only")
            if opt.freeze_doc_encoder:
                inbatch.EmbeddingModelFrozenDocEnc = inbatch_proj.EmbeddingModelFrozenDocEncProj
            else:
                # Not inbatch.EmbeddingModelDocEncNoProj = ...: inbatch.py's own __init__ calls
                # super(EmbeddingModelDocEncNoProj, self) by global name, so rebinding that global breaks
                # the base class (job 20622528, 2026-10-22). finetuning_multi instead gets a view of inbatch
                # in which only that name is swapped; inbatch's globals stay untouched.
                finetuning_multi.inbatch = _InbatchView(EmbeddingModelDocEncNoProj=inbatch_proj.EmbeddingModelDocEncProj)
        return opt


if __name__ == "__main__":
    finetuning_multi.Options = ProjOptions
    finetuning_multi.main()
