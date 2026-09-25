# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):
        # basic parameters
        self.parser.add_argument(
            "--output_dir", type=str, default="./checkpoint/", help="models are saved here"
        )
        self.parser.add_argument(
            "--run_name", type=str, default="my_experiments", help="name of the run for wandb"
        )
        self.parser.add_argument(
            "--train_data",
            nargs="+",
            default=[],
            help="Data used for training, passed as a list of directories splitted into tensor files.",
        )
        self.parser.add_argument(
            "--eval_data",
            nargs="+",
            default=[],
            help="Data used for evaluation during finetuning, this option is not used during contrastive pre-training.",
        )
        self.parser.add_argument(
            "--eval_datasets", nargs="+", default=[], help="List of datasets used for evaluation, in BEIR format"
        )
        self.parser.add_argument(
            "--eval_datasets_dir", type=str, default="./", help="Directory where eval datasets are stored"
        )
        self.parser.add_argument("--model_path", type=str, default="none", help="path for retraining")
        self.parser.add_argument("--continue_training", action="store_true")
        self.parser.add_argument("--num_workers", type=int, default=4)

        self.parser.add_argument("--chunk_length", type=int, default=256)
        self.parser.add_argument("--loading_mode", type=str, default="split")
        self.parser.add_argument("--lower_case", action="store_true", help="perform evaluation after lowercasing")
        self.parser.add_argument(
            "--sampling_coefficient",
            type=float,
            default=0.0,
            help="coefficient used for sampling between different datasets during training, \
                by default sampling is uniform over datasets",
        )
        self.parser.add_argument("--augmentation", type=str, default="none")
        self.parser.add_argument("--prob_augmentation", type=float, default=0.0)

        self.parser.add_argument("--dropout", type=float, default=0.1)
        self.parser.add_argument("--rho", type=float, default=0.05)

        self.parser.add_argument("--contrastive_mode", type=str, default="moco")
        self.parser.add_argument("--queue_size", type=int, default=65536)
        self.parser.add_argument("--temperature", type=float, default=1.0)
        self.parser.add_argument("--momentum", type=float, default=0.999)
        self.parser.add_argument("--moco_train_mode_encoder_k", action="store_true")
        self.parser.add_argument("--eval_normalize_text", action="store_true")
        self.parser.add_argument("--norm_query", action="store_true")
        self.parser.add_argument("--norm_doc", action="store_true")
        self.parser.add_argument("--projection_size", type=int, default=1536)

        self.parser.add_argument("--ratio_min", type=float, default=0.1)
        self.parser.add_argument("--ratio_max", type=float, default=0.5)
        self.parser.add_argument("--score_function", type=str, default="dot")
        self.parser.add_argument("--retriever_model_id", type=str, default="bert-base-uncased")
        self.parser.add_argument("--pooling", type=str, default="last_token")
        self.parser.add_argument("--random_init", action="store_true", help="init model with random weights")

        # dataset parameters
        self.parser.add_argument("--per_gpu_batch_size", default=64, type=int, help="Batch size per GPU for training.")
        self.parser.add_argument(
            "--per_gpu_eval_batch_size", default=256, type=int, help="Batch size per GPU for evaluation."
        )
        self.parser.add_argument("--total_steps", type=int, default=1000)
        self.parser.add_argument("--warmup_steps", type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=10001, help="Master port (for multi-node SLURM jobs)")
        self.parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
        # training parameters
        self.parser.add_argument("--optim", type=str, default="adamw")
        self.parser.add_argument("--scheduler", type=str, default="linear")
        self.parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
        self.parser.add_argument(
            "--lr_min_ratio",
            type=float,
            default=0.0,
            help="minimum learning rate at the end of the optimization schedule as a ratio of the learning rate",
        )
        self.parser.add_argument("--weight_decay", type=float, default=0.01, help="learning rate")
        self.parser.add_argument("--beta1", type=float, default=0.9, help="beta1")
        self.parser.add_argument("--beta2", type=float, default=0.98, help="beta2")
        self.parser.add_argument("--eps", type=float, default=1e-6, help="eps")
        self.parser.add_argument(
            "--log_freq", type=int, default=100, help="log train stats every <log_freq> steps during training"
        )
        self.parser.add_argument(
            "--eval_freq", type=int, default=500, help="evaluate model every <eval_freq> steps during training"
        )
        self.parser.add_argument("--save_freq", type=int, default=50000)
        self.parser.add_argument("--maxload", type=int, default=None)
        self.parser.add_argument("--label_smoothing", type=float, default=0.0)
        self.parser.add_argument("--accumulation_steps", type=int, default=1)

        # finetuning options
        self.parser.add_argument("--negative_ctxs", type=int, default=1)
        self.parser.add_argument("--negative_hard_min_idx", type=int, default=0)
        self.parser.add_argument("--negative_hard_ratio", type=float, default=0.0)
        
        # added 
        self.parser.add_argument("--training_mode", type=str, default="standard_org_q")
        self.parser.add_argument(
            "--loss_fn",
            type=str,
            default="auto",
            choices=["auto", "contrastive", "hungarian_masked", "hungarian"],
            help=(
                "Loss used by EmbeddingModelDocEncNoProj. 'auto' picks "
                "hungarian_masked when training_mode=='multi' and contrastive "
                "otherwise. 'hungarian' is the legacy HungarianContrastiveLoss "
                "(kept for comparison; has the same-example false-negative issue)."
            ),
        )
        self.parser.add_argument("--eval_recall", action="store_true")
        self.parser.add_argument("--sample_length", action="store_true")
        self.parser.add_argument("--max_positive_documents", type=int, default=1)
        self.parser.add_argument("--not_save", action='store_true')
        self.parser.add_argument(
            "--save_every_eval",
            action="store_true",
            help=(
                "If set, save a step-numbered checkpoint (checkpoint/step-<N>) "
                "at every eval_freq, independent of the best-so-far MRR gate "
                "used for the 'best_model' checkpoint. Needed to evaluate the "
                "full training trajectory rather than only whichever eval "
                "happened to be 'best' at the time (which resets to 0 on every "
                "restart, since best_eval_metric is re-initialized each run -- "
                "see experiment_plan.md's multi_hungarian_with_detach writeup). "
                "Saves far more checkpoints than typically needed for eval -- "
                "consider --save_at_steps instead if disk space is tight."
            ),
        )
        self.parser.add_argument(
            "--save_at_steps",
            nargs="+",
            type=int,
            default=None,
            help=(
                "If set, save a step-numbered checkpoint (checkpoint/step-<N>) "
                "at exactly these training steps, independent of the "
                "best-so-far MRR gate. Same purpose as --save_every_eval but "
                "bounded disk cost -- use this when only specific steps will "
                "actually be evaluated."
            ),
        )
        self.parser.add_argument("--doc_lengths", nargs='+', default=[3])
        self.parser.add_argument(
            "--full_sampling",
            action="store_true",
            help=(
                "If set, sampling_rate=1.0 always (fully autoregressive, no teacher forcing). "
                "Default is the linear ramp sampling_rate=step/total_steps."
            ),
        )
        self.parser.add_argument(
            "--freeze_doc_encoder",
            action="store_true",
            help=(
                "If set, use EmbeddingModelFrozenDocEnc(SingleQuery) instead of "
                "EmbeddingModelDocEncNoProj(SingleQuery): documents are encoded by a "
                "second, separate, frozen copy of the backbone (matching the paper's "
                "frozen-document-encoder design) instead of the trainable query-encoder "
                "weights. Does not modify EmbeddingModelDocEncNoProj."
            ),
        )
        self.parser.add_argument(
            "--force_causal",
            action="store_true",
            help=(
                "Only used with --freeze_doc_encoder and training_mode=multi "
                "(EmbeddingModelFrozenDocEnc). If set, passes is_causal=True on every "
                "query-encoder forward call, matching the paper's autoregressive-decoder-LM "
                "assumption. Default is bidirectional (is_causal=False), the backbone's "
                "normal embedding-model behavior."
            ),
        )
        self.parser.add_argument(
            "--use_lora",
            action="store_true",
            help=(
                "Only used with --freeze_doc_encoder (EmbeddingModelFrozenDocEnc(SingleQuery)). "
                "If set, wraps the trainable query encoder in a LoRA adapter (peft) instead of "
                "full fine-tuning, matching the paper's real-data recipe (Appendix A.6). The "
                "frozen doc_encoder never gets LoRA -- it isn't trained at all either way. "
                "Isolates whether full-FT-without-LoRA is itself degrading retrieval quality, "
                "independent of the multi-query objective (see experiment_plan.md)."
            ),
        )
        self.parser.add_argument("--lora_r", type=int, default=64)
        self.parser.add_argument("--lora_alpha", type=int, default=16)
        self.parser.add_argument("--lora_dropout", type=float, default=0.1)

    def print_options(self, opt):
        message = ""
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: %s]" % str(default)
            message += f"{str(k):>40}: {str(v):<40}{comment}\n"
        print(message, flush=True)
        model_dir = os.path.join(opt.output_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(os.path.join(opt.output_dir, "models"))
        file_name = os.path.join(opt.output_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):
        opt, _ = self.parser.parse_known_args()
        # opt = self.parser.parse_args()
        return opt
