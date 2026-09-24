#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=retrieve_ambigqa_ddp_step750
#SBATCH --output=sbatch_outputs/retrieve_ambigqa_ddp_step750.out
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1

DATA_NAME="ambigqa"
DATA_PATH="data/amer_data/eval_data/${DATA_NAME}.jsonl"
PASSAGES="/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv"
PASSAGES_EMBEDDINGS="wikipedia_embeddings/ambigqa/ddp_step750/*"
MODEL_NAME_OR_PATH="checkpoints/ambigqa/ambigqa_joint_repro_steps800_t0.05_lr0.00001_ws30_bs50_hungarian_ddp/checkpoint/step-750/"
OUTPUT_DIR="results/finetuned/ambigqa/ddp_step750/"
OUTPUT_FILE="ambigqa.jsonl"
PROJECTION_SIZE=1536
N_DOCS=500
NUM_SHARDS=16
PER_GPU_BATCH_SIZE=4

singularity exec --nv \
            --overlay /scratch/hc3337/envs/nli.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; python retrieval_inf.py \
                --model_name_or_path $MODEL_NAME_OR_PATH \
                --passages $PASSAGES \
                --passages_embeddings '$PASSAGES_EMBEDDINGS' \
                --data $DATA_PATH \
                --output_dir $OUTPUT_DIR \
                --projection_size $PROJECTION_SIZE \
                --per_gpu_batch_size $PER_GPU_BATCH_SIZE \
                --n_docs $N_DOCS \
                --num_shards $NUM_SHARDS \
                --use_gpu \
                --output_file $OUTPUT_FILE \
                --max_new_tokens 2"
