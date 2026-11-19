#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=96GB
#SBATCH --account=torch_pr_152_echoi
#SBATCH --constraint=h200|l40s|b200
#SBATCH --partition=h200_courant,l40s_courant,h200_public,l40s_public,h200,b200
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --gres=gpu:1
#SBATCH --job-name=f0_bounds
#SBATCH --output=sbatch_outputs/f0_bounds.out
# F0 headroom bounds + gold geometry in the frozen base document space (tools/f0_bounds.py).
singularity exec --nv \
            --overlay /scratch/hc3337/envs/nli.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; python training/inf_retriever/tools/f0_bounds.py --out ${OUT:-results/phaseC/frozen_f/f0}"
