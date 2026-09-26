#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --tasks-per-node=1
#SBATCH --time=1:00:00
#SBATCH --mem=64GB
#SBATCH --account=torch_pr_152_courant
#SBATCH --job-name=eval_drift_ckpt
#SBATCH --output=sbatch_outputs/evaldrift_%x.out

# CPU job: score a retrieval output (MRecall/Recall @100/@10) into OUT_DIR/eval_metrics.txt
# and append the checkpoint's parameter drift vs its run's step-0 to DRIFT_OUT.
#   sbatch --job-name=<tag> --export=ALL,DATA_NAME=...,OUT_DIR=...,RUN_DIR=...,STEP=...,DRIFT_OUT=... \
#       training/inf_retriever/tools/eval_drift_ckpt.sbatch
: "${DATA_NAME:?}" "${OUT_DIR:?}" "${RUN_DIR:?}" "${STEP:?}" "${DRIFT_OUT:?}"
GOLD_ARG=""
[ "$DATA_NAME" = "ambigqa" ] && GOLD_ARG="--no-gold-id"

singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; \
                python eval.py --data_path data/amer_data/eval_data/${DATA_NAME}.jsonl --topk 100 10 $GOLD_ARG \
                    --input-file $OUT_DIR/${DATA_NAME}.jsonl > $OUT_DIR/eval_metrics.txt 2>&1 && \
                python training/inf_retriever/tools/param_drift.py --run_dir $RUN_DIR --steps $STEP --out $DRIFT_OUT"
