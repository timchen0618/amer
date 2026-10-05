#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:00:00
#SBATCH --mem=48GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --job-name=kagg_eval
#SBATCH --output=sbatch_outputs/%x.out
# Score the per-step lists of kagg_retrieve.sbatch for every k in KS and both aggregations.
#   sbatch --job-name=<tag> --export=ALL,DS=qampari,DEV=<dev jsonl>,KS="1 2 ... 10",OUT_DIR=<dir>[,SETS="dev test"] training/inf_retriever/tools/kagg_eval.sbatch
: "${DS:?}" "${DEV:?}" "${KS:?}" "${OUT_DIR:?}"
SETS=${SETS:-dev test}
for set in $SETS; do
  if [ $set = dev ]; then DATA=$DEV; else DATA=data/amer_data/eval_data/$DS.jsonl; fi
  singularity exec --overlay /scratch/hc3337/envs/nli.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; python training/inf_retriever/tools/kagg_eval.py \
      --ds $DS --per_step $OUT_DIR/${set}_steps.jsonl --data $DATA --ks $KS --aggs round_robin rrf --out $OUT_DIR/${set}_kagg.json" || exit 1
done
echo ALL_DONE
