#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=4-00:00:00
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --partition=h200_courant,h200_public
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/hc3337/projects/autoregressive/paper_repro_data/logs/train_%x_%j.out
# Paper-repro training (fe0bd78 train_distributed.py, unchanged). usage:
#   sbatch --job-name=repro_<lm>_<ds>_<cond> train.sbatch <lm> <ds: qampari|ambiguous_qe> <cond: single|multi_SS|multi_sampling> [extra args]
# Paper args per wandb configs of the 2025-10 seeded runs; changed per user: batch 50/GPU x 2 GPUs, srm 1, seed 0 only.
set -euo pipefail
LM=$1; DS=$2; COND=$3; shift 3; EXTRA="$*"
W=/scratch/hc3337/projects/autoregressive/.claude/worktrees/paper-repro
D=/scratch/hc3337/projects/autoregressive/paper_repro_data
case $LM in llama-1b) MID=meta-llama/Llama-3.2-1B-Instruct;; llama-3b) MID=meta-llama/Llama-3.2-3B-Instruct;;
  llama-8b) MID=meta-llama/Llama-3.1-8B-Instruct;; qwen3-4b) MID=Qwen/Qwen3-4B-Instruct-2507;; esac
if [ $DS = qampari ]; then LR=5e-5; SAVE=500; RANGE=5_to_8; else LR=2e-4; SAVE=200; RANGE=2_to_5; fi
case $COND in
  single)         C="--loss_function Contrastive --shuffle_sequence --take_first";;
  multi_SS)       C="--loss_function Hungarian_Contrastive --shuffle_sequence --mix_one_label_shuffled";;
  multi_sampling) C="--loss_function Hungarian_Contrastive --shuffle_sequence --mix_one_label_shuffled --force_sampling";;
esac
NAME=${LM}_${COND}_0
ARGS="--project diverse_retrieval_repro --save_path $D/results/$LM/${DS}_inf --name $NAME \
  --train_path $D/training_datasets/$LM/$DS/inf/autoregressive_${DS}_inf_train_dataset_1b_contrastive_${RANGE}_ctxs/ \
  --adapter_path None --linear_checkpoint_path None --model_id $MID \
  --lr $LR --temperature 0.05 --batch_size_training 50 --num_epochs 120 --warmup_ratio 0.05 --lr_min_ratio 0.0 \
  --gradient_accumulation_steps 1 --weight_decay 0.01 --scheduler linear --max_grad_norm 5.0 \
  --embedding_model_dim 1536 --save_every_n_steps $SAVE --model_type EmbeddingModelSSVariableLeftPad \
  --save_only_improve --save_best_model --schedule_sampling --left_padding --normalize_embeddings \
  --sample_rate_multiplier 1 --log_with wandb --seed 0 $C $EXTRA"
echo "host $(hostname) start $(date) args: $ARGS"
mkdir -p $D/results/$LM/${DS}_inf
singularity exec --nv --overlay /scratch/hc3337/envs/div.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  bash -c "source /ext3/env.sh; cd $W; accelerate launch --config_file $W/scripts/paper_repro/accelerate_fsdp_2gpu.yaml train_distributed.py $ARGS"
echo "end $(date)"
