#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=kmeans_eli5
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --output=kmeans_eli5.out
#SBATCH --gres=gpu:1
#SBATCH --partition=h200
#SBATCH --array=0-49

SINGULARITY_IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/nli.ext3


singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive/; (trap 'kill 0' SIGINT; HF_TOKEN=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 python large_scale/generate_data.py --model_name infly/inf-retriever-v1-1.5b --input_file eli5_inf+contriever_top500_split_$SLURM_ARRAY_TASK_ID.jsonl --retrieval_results_dir /scratch/hc3337/projects/autoregressive/large_scale/data/eli5/ --output_dir /scratch/hc3337/projects/autoregressive/large_scale/clustered_data/eli5/ & wait)" 
