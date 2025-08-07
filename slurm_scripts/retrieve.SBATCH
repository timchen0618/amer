#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --mem=300GB
#SBATCH --job-name=run_eval
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --output=run_eval.out
#SBATCH --gres=gpu:1

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/div.ext3

DATA_NAME="ambiguous_qe"
SPLIT="dev"
SUFFIX_LIST="_contrastive_4_gpus_lr2e5_ep20_temp0.05_warmup0.05"
RETRIEVER_LIST="inf"
NUM_SHARDS="16"
CHECKPOINT_NUM="30000"

singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive/; (trap 'kill 0' SIGINT; HF_TOKEN=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 python gen_ret_and_eval.py --data_name $DATA_NAME --suffix_list $SUFFIX_LIST --retriever_list $RETRIEVER_LIST --use_gpu --num_shards $NUM_SHARDS --use_best_model --checkpoint_num $CHECKPOINT_NUM --compute_loss  & wait)" 
