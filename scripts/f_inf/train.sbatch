#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=2-00:00:00
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --partition=h200_courant,h200_public
#SBATCH --gres=gpu:2
#SBATCH --output=/scratch/hc3337/projects/autoregressive/f_paper_data/logs/train_%x_%j.out
# F paper-stack runs P1/P2 (reports/objective_audit_2026-10-13/F_paper_stack_runs_2026-10-27.md): R's launcher
# (scripts/paper_repro/train.sbatch) with INF-Retriever as the query model. usage:
#   sbatch --job-name=f_<P1|P2>_<cond> train.sbatch <P1|P2|SMOKE> <cond: single|multi_SS> [extra args]
# Paper args as R's runs: batch 50/GPU x 2 GPUs, srm 1, seed 0. P2 adds --valid_path (the seeded split's dev set).
set -euo pipefail
RUN=$1; COND=$2; shift 2; EXTRA="$*"
W=/scratch/hc3337/projects/autoregressive/.claude/worktrees/f-paper-inf
D=/scratch/hc3337/projects/autoregressive/f_paper_data
MID=infly/inf-retriever-v1-1.5b
case $RUN in
  P1) TP=$D/training_datasets/inf/qampari/inf/autoregressive_qampari_inf_train_dataset_1b_contrastive_5_to_8_ctxs/; VP="";;
  P2) TP=$D/training_datasets/inf/qampari_s0/inf/autoregressive_qampari_s0_inf_train_dataset_1b_contrastive_5_to_8_ctxs/
      VP="--valid_path $D/training_datasets/inf/qampari_s0dev/inf/autoregressive_qampari_s0dev_inf_train_dataset_1b_contrastive_5_to_8_ctxs/";;
  SMOKE) TP=/scratch/hc3337/projects/autoregressive/training_datasets/inf/qampari/inf/autoregressive_qampari_inf_train_dataset_1b_contrastive_5_to_8_ctxs/; VP="";;
esac
case $COND in
  single)   C="--loss_function Contrastive --shuffle_sequence --take_first";;
  multi_SS) C="--loss_function Hungarian_Contrastive --shuffle_sequence --mix_one_label_shuffled";;
esac
NAME=${RUN}_inf_${COND}_0
ARGS="--project amer_f_paper_inf --save_path $D/results/inf/qampari_inf --name $NAME \
  --train_path $TP $VP \
  --adapter_path None --linear_checkpoint_path None --model_id $MID \
  --lr 5e-5 --temperature 0.05 --batch_size_training 50 --num_epochs ${EPOCHS:-120} --warmup_ratio 0.05 --lr_min_ratio 0.0 \
  --gradient_accumulation_steps 1 --weight_decay 0.01 --scheduler linear --max_grad_norm 5.0 \
  --embedding_model_dim 1536 --save_every_n_steps 500 --model_type EmbeddingModelSSVariableLeftPad \
  --save_only_improve --save_best_model --schedule_sampling --left_padding --normalize_embeddings \
  --sample_rate_multiplier 1 --log_with wandb --seed 0 --gradient_checkpointing $C $EXTRA"
echo "host $(hostname) start $(date) args: $ARGS"
mkdir -p $D/results/inf/qampari_inf
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -l 60 > $D/logs/gpumem_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.csv &
singularity exec --nv --overlay /scratch/hc3337/envs/div.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  bash -c "source /ext3/env.sh; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd $W; accelerate launch --config_file $W/scripts/paper_repro/${ACC_CFG:-accelerate_fsdp_2gpu.yaml} train_distributed.py $ARGS"
echo "end $(date)"
