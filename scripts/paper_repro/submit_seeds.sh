#!/bin/bash
# Submit seed runs for the headline settings plus per-LM eval/score chains.
# usage: bash submit_seeds.sh <ds: ambiguous_qe|qampari> <seed> [seed ...]
# AmbigQA: single + multi_SS (multiplier 1). QAMPARI: single + multi_SS multiplier 10. Recipe otherwise = seed 0.
# Evals for one LM run one seed after another: each rewrites the shared top500_*.pkl.
set -euo pipefail
DS=$1; shift; SEEDS="$*"
S=$(dirname "$0"); LOG=/scratch/hc3337/projects/autoregressive/paper_repro_data/logs/jobs_seeds.txt
if [ $DS = qampari ]; then
  MC=multi_SS_srm10; MARGS="--sample_rate_multiplier 10"
  declare -A TS=([llama-1b]=08:00:00 [llama-3b]=12:00:00 [qwen3-4b]=16:00:00 [llama-8b]=20:00:00)
  declare -A TM=([llama-1b]=24:00:00 [llama-3b]=40:00:00 [qwen3-4b]=48:00:00 [llama-8b]=72:00:00)
else
  MC=multi_SS; MARGS=""
  declare -A TS=([llama-1b]=12:00:00 [llama-3b]=12:00:00 [qwen3-4b]=12:00:00 [llama-8b]=12:00:00)
  declare -A TM=([llama-1b]=12:00:00 [llama-3b]=12:00:00 [qwen3-4b]=12:00:00 [llama-8b]=12:00:00)
fi
for lm in llama-1b llama-3b qwen3-4b llama-8b; do
  prev=""
  for sd in $SEEDS; do
    a=$(sbatch --parsable --time=${TS[$lm]} --job-name=repro_${lm}_${DS}_single_s$sd $S/train.sbatch $lm $DS single --seed $sd --name ${lm}_single_$sd)
    b=$(sbatch --parsable --time=${TM[$lm]} --job-name=repro_${lm}_${DS}_${MC}_s$sd $S/train.sbatch $lm $DS multi_SS $MARGS --seed $sd --name ${lm}_${MC}_$sd 2>/dev/null)
    echo "$lm $DS single seed$sd train $a" >> $LOG; echo "$lm $DS $MC seed$sd train $b" >> $LOG
    dep="afterok:$a:$b"; [ -n "$prev" ] && dep="$dep:$prev"
    e=$(SEED=$sd sbatch --parsable --dependency=$dep --job-name=repro_eval_${lm}_${DS}_s$sd $S/eval_embed_search.sbatch $lm $DS single $MC); prev=$e
    echo "$lm $DS seed$sd eval $e (dep $dep)" >> $LOG
  done
  s=$(sbatch --parsable --dependency=afterok:$prev --job-name=repro_score_${lm}_$DS $S/eval_score.sbatch $lm $DS full)
  o=$(sbatch --parsable --dependency=afterok:$prev --job-name=repro_oldsearch_${lm}_$DS $S/search_old_index.sbatch $lm/$DS)
  so=$(sbatch --parsable --dependency=afterok:$o --job-name=repro_score_old_${lm}_$DS $S/eval_score.sbatch $lm $DS old21m)
  echo "$lm $DS score $s oldsearch $o oldscore $so" >> $LOG
done
