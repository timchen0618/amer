#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=24:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=clustering_dbscan
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --output=clustering_dbscan.out
#SBATCH --partition=h200
#SBATCH --gres=gpu:1

SINGULARITY_IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/nli.ext3

singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive/large_scale/ ; (trap 'kill 0' SIGINT; HF_TOKEN=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 python generate_clustered_data.py --model_name infly/inf-retriever-v1-1.5b --input_file eli5+researchy_questions_1k.jsonl  --retrieval_results_dir data/ --output_dir clustered_data/ --save_embeddings --method dbscan & wait)" 
