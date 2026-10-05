#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200|l40s|b200
#SBATCH --partition=h200_courant,l40s_courant,h200_public,l40s_public,b200,h200
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --gres=gpu:1
#SBATCH --job-name=kagg_retrieve
#SBATCH --output=sbatch_outputs/%x.out
# C2 / aggregation study: retrieve once with KMAX query embeddings per question and save every
# embedding's own top-500 list (retrieval_inf.py --save_per_step), for the dev and test sets.
# tools/kagg_eval.py then scores every k <= KMAX and both aggregations offline.
#   sbatch --job-name=<tag> --export=ALL,CKPT=<ckpt dir>,EMB_DIR=<corpus embeddings>,DS=qampari,DEV=<dev jsonl>,KMAX=10,OUT_DIR=<dir>[,CORPUS=<tsv>,NUM_SHARDS=16] \
#       training/inf_retriever/tools/kagg_retrieve.sbatch
: "${CKPT:?}" "${EMB_DIR:?}" "${DS:?}" "${DEV:?}" "${KMAX:?}" "${OUT_DIR:?}"
CORPUS=${CORPUS:-/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv}; NUM_SHARDS=${NUM_SHARDS:-16}
mkdir -p "$OUT_DIR"
for set in ${SETS:-dev test}; do
  if [ $set = dev ]; then DATA=$DEV; else DATA=data/amer_data/eval_data/$DS.jsonl; fi
  singularity exec --nv --overlay /scratch/hc3337/envs/nli.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; python retrieval_inf.py \
      --model_name_or_path $CKPT --passages $CORPUS --passages_embeddings '$EMB_DIR/*' --data $DATA \
      --output_dir $OUT_DIR --projection_size 1536 --per_gpu_batch_size 4 --n_docs 500 --num_shards $NUM_SHARDS \
      --use_gpu --max_new_tokens $KMAX --save_per_step $OUT_DIR/${set}_steps.jsonl" || exit 1
done
echo ALL_DONE
