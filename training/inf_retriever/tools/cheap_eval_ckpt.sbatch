#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH --job-name=cheap_eval
#SBATCH --output=sbatch_outputs/cheap_eval_%x.out

# Phase A cheap eval: the same embed -> retrieve -> eval pipeline as the full-corpus
# evaluation, but over a reduced corpus (data/phaseA/corpora/<SET>.tsv: base top-1000
# candidates + golds + 100k random passages). Embeddings (~3 GB) are deleted at the end.
#   sbatch --job-name=<tag> --export=ALL,CKPT=<checkpoint dir or HF id>,SET=<qampari_test|ambigqa_test|qampari_dev500|ambigqa_dev300>,DATA=<eval jsonl>,OUT_DIR=<dir>,K=<k or empty> \
#       training/inf_retriever/tools/cheap_eval_ckpt.sbatch
# A HF model id (e.g. infly/inf-retriever-v1-1.5b) is evaluated with retrieval_base.py, as
# the existing base-model results were.
: "${CKPT:?}" "${SET:?}" "${DATA:?}" "${OUT_DIR:?}"
cd /scratch/hc3337/projects/autoregressive
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
CORPUS=data/phaseA/corpora/${SET}.tsv
EMB=wikipedia_embeddings/phaseA/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
DS=${SET%%_*}                      # qampari | ambigqa
GOLD_ARG=""; [ "$DS" = "ambigqa" ] && GOLD_ARG="--no-gold-id"
K_ARG="";    [ -n "${K:-}" ] && K_ARG="--max_new_tokens ${K}"
# Passage truncation (code audit 2026-09-28, R3): 512 tokens = training's --chunk_length. The 24-run
# grid's checkpoints (grid_*) were all evaluated at 1024, the old hard-coded value, and keep it so
# their results stay comparable. PASSAGE_MAXLEN overrides both.
case "$CKPT" in */grid_*|grid_*) PML_DEFAULT=1024 ;; *) PML_DEFAULT=512 ;; esac
PML=${PASSAGE_MAXLEN:-$PML_DEFAULT}
echo "passage max length $PML tokens"
mkdir -p "$EMB" "$OUT_DIR"
trap 'rm -rf "$EMB"' EXIT

singularity exec --nv --overlay /scratch/hc3337/envs/div.ext3:ro $IMG bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; \
    python gen_embed_new.py --passages $CORPUS --output_dir $EMB --shard_id 0 --num_shards 1 --model_name_or_path $CKPT --per_gpu_batch_size 512 --passage_maxlength $PML" || exit 1

if [[ "$CKPT" == */* && ! -e "$CKPT" ]]; then RETRIEVER=retrieval_base.py; K_ARG=""; else RETRIEVER=retrieval_inf.py; fi
singularity exec --nv --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG bash -c "source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; \
    python $RETRIEVER --model_name_or_path $CKPT --passages $CORPUS --passages_embeddings '$EMB/*' --data $DATA \
        --output_dir $OUT_DIR --projection_size 1536 --per_gpu_batch_size 4 --n_docs 500 --num_shards 1 --use_gpu \
        --output_file results.jsonl $K_ARG" || exit 1

singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro $IMG bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; \
    python eval.py --data_path $DATA --topk 100 10 $GOLD_ARG --input-file $OUT_DIR/results.jsonl" > $OUT_DIR/eval_metrics.txt 2>&1
grep MRecall $OUT_DIR/eval_metrics.txt
