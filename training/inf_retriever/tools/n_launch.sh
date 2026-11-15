#!/bin/bash
# Session N (2026-11-12): launch one F-style frozen QAMPARI S0 run with the worktree's f_run.sh (CODE_ROOT = this
# worktree; data, checkpoints and results on the main checkout). Same recipe and env as F's f_queue.sh entries
# (BASE + S0 + S0ENV + LONG20), plus the retriever:
#   bash n_launch.sh <run suffix> <inf|nemotron|qwen3> "<extra flags>"
# e.g. bash n_launch.sh nemo_S0_cap10k_onelabel nemotron "--loss_fn hungarian --sampling_saturate_frac 0.1 --sampling_cap 0.8 --one_label_step"
# Runs whose suffix contains "single" use mode single. Run name: clean_v3_n_qampari_<suffix> (not counted by f_queue).
# SMOKE=1: 30 steps, save/eval at 30, 1 h walltime, run name with a _smoke suffix.
# N_LR (default 1e-5): learning rate. N_DATA_ROOT (default data/training/clean_v3_hn_s0): training data root (its
# qampari/train_data.jsonl trains, its qampari/dev_data.jsonl is the in-training eval set), e.g. the
# Nemotron-mined hard negatives data/training/clean_v3_hn_s0_nemo (2026-11-13). Full dev / test unchanged.
set -eu
WT=/scratch/hc3337/projects/autoregressive/.claude/worktrees/n-nemotron
cd /scratch/hc3337/projects/autoregressive
SUF=$1; RET=$2; FLAGS=${3:-}
run=clean_v3_n_qampari_$SUF; [ -n "${SMOKE:-}" ] && run=${run}_smoke
mode=multi; case $SUF in *single*) mode=single ;; esac
DROOT=${N_DATA_ROOT:-data/training/clean_v3_hn_s0}; LR=${N_LR:-1e-5}
[ -s $DROOT/qampari/train_data.jsonl ] || { echo "missing $DROOT/qampari/train_data.jsonl"; exit 1; }
S0="--train_data $DROOT/qampari/train_data.jsonl"
BASE="--train_data data/training/clean_v3_hn/qampari/train_data.jsonl --negative_hard_ratio 0.5 --freeze_doc_encoder"
export F_DEV=data/phaseA/qampari_clean_v3_dev_s0.jsonl F_DATA_ROOT=$DROOT
if [ -n "${SMOKE:-}" ]; then
  export STEPS_OVERRIDE=30 WARMUP_OVERRIDE=5 FREQ_OVERRIDE=30 F_STEPS=30 F_FINAL=30 TRAIN_TIME=01:00:00
else
  export STEPS_OVERRIDE=10000 WARMUP_OVERRIDE=400 F_STEPS=1000,2000,2500,3000,4000,5000,6000,7000,8000,9000,10000 \
         F_FINAL=10000 TRAIN_TIME=20:00:00
fi
export CODE_ROOT=$WT TRAIN_SCRIPT=finetuning_multi_nemo.py
if [ "$RET" = nemotron ]; then
  export TRAIN_OVERLAY=nemotron RET_ENV=nemotron PROJ_SIZE=2048 F_EMB=wikipedia_embeddings/qampari/nemotron_frozen_base_512
  FLAGS="$FLAGS --retriever nemotron"
elif [ "$RET" = qwen3 ]; then  # 2026-11-15: Qwen3-Embedding-0.6B; div.ext3 trains, nli retrieves (as inf-retriever)
  export PROJ_SIZE=1024 F_EMB=wikipedia_embeddings/qampari/qwen3emb06_frozen_base_512
  FLAGS="$FLAGS --retriever qwen3"
fi
setsid nohup bash $WT/training/inf_retriever/tools/f_run.sh $mode $LR $run "$BASE $S0 $FLAGS" \
  >> results/phaseC/$run.driver.out 2>&1 < /dev/null &
echo "started $run (driver pid $!, host $(hostname -s))"
