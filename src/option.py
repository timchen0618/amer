import argparse

def get_training_args():
    """
    Create and return argument parser for training configuration.
    This is shared between single GPU and distributed training scripts.
    """
    parser = argparse.ArgumentParser(description="Train autoregressive model")
    
    # Main configuration
    parser.add_argument("--project", type=str, default="diverse_retrieval", help="Project name")
    parser.add_argument("--save_path", type=str, default="results/ambiguous_qe_inf/", help="Directory to save results")
    parser.add_argument("--name", type=str, default="toy_contrastive_4_gpus_from_stage2_lr2e5_ep20_temp0.05_warmup0.05", help="Experiment name")
    parser.add_argument("--train_path", type=str, default="training_datasets/ambiguous_qe/inf/autoregressive_ambiguous_qe_inf_train_dataset_1b_contrastive_2_to_5_ctxs/", help="Path to training dataset")
    
    # Save and load configuration
    parser.add_argument("--save_every_n_steps", type=int, default=50, help="Save model every n steps")
    parser.add_argument("--save_best_model", action="store_true", default=False, help="Save best model")
    parser.add_argument("--embedding_model_dim", type=int, default=1536, help="Embedding model dimension")
    parser.add_argument("--adapter_path", type=str, default="results/nq_inf/toy_contrastive/checkpoint_70000", help="Path to adapter checkpoint")
    parser.add_argument("--linear_checkpoint_path", type=str, default="results/nq_inf/toy_contrastive/checkpoint_70000_linear.pt", help="Path to linear checkpoint")
    # adapter_path: results/qampari_inf/toy_qemb_from_nq/checkpoint_30000
    # linear_checkpoint_path: results/qampari_inf/toy_qemb_from_nq/checkpoint_30000_linear.pt
    # adapter_path: 
    # linear_checkpoint_path: 
    
    
    # Training configuration
    parser.add_argument("--batch_size_training", type=int, default=16, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
    parser.add_argument("--scheduler", type=str, default="linear", help="Learning rate scheduler")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")
    
    # Training options
    parser.add_argument("--shuffle_sequence", action="store_true", default=False, help="Shuffle sequence during training")
    parser.add_argument("--train_on_all_data", action="store_true", default=False, help="Train on all available data")
    parser.add_argument("--save_only_improve", action="store_true", default=False, help="Save only when validation improves")
    parser.add_argument("--take_first", action="store_true", default=False, help="Take first sequence")
    
    # Model architecture
    parser.add_argument("--temperature", type=float, default=0.05, help="Temperature for contrastive loss")
    parser.add_argument("--loss_function", type=str, default="Hungarian_Contrastive", 
                       choices=["MSE", "Hungarian_MSE", "Contrastive", "Hungarian_Contrastive"],
                       help="Loss function to use")
    parser.add_argument("--extra_q_embed", action="store_true", default=False, help="Use extra question embedding")
    parser.add_argument("--compute_loss_on_q", action="store_true", default=False, help="Compute loss on questions")
    parser.add_argument("--use_eos", action="store_true", default=False, help="Use EOS token")
    
    # Data loading
    parser.add_argument("--first_label_only", action="store_true", default=False, help="Use first label (question) only")
    
    # Debug and advanced options
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode. Not saving model.")
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Model ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", type=int, default=0, help="Resume from epoch")
    
    # Optimizer configuration
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta2")
    parser.add_argument("--optim", type=str, default="adamw", help="Optimizer type")
    parser.add_argument("--lr_min_ratio", type=float, default=0.0, help="Minimum learning rate ratio")
    parser.add_argument("--eps", type=float, default=1e-6, help="Optimizer epsilon")
    parser.add_argument("--weight_tying", action="store_true", default=False, help="Use weight tying")
    
    return parser
