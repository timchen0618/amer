#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=run_train_gaussian
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --output=run_train_gaussian.out
#SBATCH --gres=gpu:1

SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/div.ext3


singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive/; (trap 'kill 0' SIGINT; HF_TOKEN=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 python train.py & wait)" 
