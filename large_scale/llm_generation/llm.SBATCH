#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=8:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=llm_generation_q2docs_existing_2_150
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --output=llm_generation_q2docs_existing_2_150.out
#SBATCH --gres=gpu:a100:1

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/nli.ext3

singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive/large_scale/llm_generation/ ; (trap 'kill 0' SIGINT; HF_TOKEN=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 python generate.py --command existing_q2docs_2 --output_dir vllm_outputs/q_docs_existing_2/ --question_source existing --length 150 & wait)" 
