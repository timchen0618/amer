#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=8:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=run_reranking_ambiguous_qe_contrastive_all_labels_ordered
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --output=run_reranking_ambiguous_qe_contrastive_all_labels_ordered.out
#SBATCH --gres=gpu:1
#SBATCH --constraint="h200|l40s"
#SBATCH --comment="preemption=yes;requeue=yes"

machine="torch"

if [ "$machine" == "torch" ]; then
    SINGULARITY_IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
    OVERLAY_FILE=/scratch/hc3337/envs/nli.ext3
elif [ "$machine" == "greene" ]; then
    SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
    OVERLAY_FILE=/scratch/hc3337/envs/nli.ext3
fi

singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive/baselines/reranking/ ; (trap 'kill 0' SIGINT; HF_TOKEN=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 python gen_doc_similarity.py --base_retriever ambiguous_qe_contrastive_all_labels_ordered & wait)" 
