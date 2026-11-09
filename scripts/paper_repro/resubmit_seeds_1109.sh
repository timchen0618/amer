#!/bin/bash
# 2026-11-09: /scratch quota filled at 04:58 and killed every running/queued seed job.
# Retrain the unfinished runs, then re-chain evals + scores for all LMs.
# llama-3b qampari multi s1 was in its last epoch (119.4/120) -> keep its best-dev checkpoint.
set -euo pipefail
S=$(dirname "$0"); LOG=/scratch/hc3337/projects/autoregressive/paper_repro_data/logs/jobs_seeds.txt
echo "# resubmit 2026-11-09 after quota failure" >> $LOG
trn() { # lm ds cond seed time extra...
  local lm=$1 ds=$2 c=$3 sd=$4 t=$5; shift 5
  local name=${lm}_${c}_$sd; [ $c = multi_SS ] && [ $ds = qampari ] && name=${lm}_multi_SS_srm10_$sd
  sbatch --parsable --time=$t --job-name=repro_${lm}_${ds}_${name#${lm}_}_r $S/train.sbatch $lm $ds $c "$@" --seed $sd --name $name
}
declare -A DEP
DEP[llama-3b,qampari,2]=$(trn llama-3b qampari multi_SS 2 40:00:00 --sample_rate_multiplier 10)
DEP[qwen3-4b,qampari,1]=$(trn qwen3-4b qampari multi_SS 1 48:00:00 --sample_rate_multiplier 10)
DEP[qwen3-4b,qampari,2]=$(trn qwen3-4b qampari multi_SS 2 48:00:00 --sample_rate_multiplier 10)
DEP[llama-8b,qampari,1]=$(trn llama-8b qampari single 1 20:00:00)
DEP[llama-8b,qampari,2]=$(trn llama-8b qampari single 2 20:00:00)
for k in "${!DEP[@]}"; do echo "train $k ${DEP[$k]}" >> $LOG; done
for DS in ambiguous_qe qampari; do
  [ $DS = qampari ] && MC=multi_SS_srm10 || MC=multi_SS
  for lm in llama-1b llama-3b qwen3-4b llama-8b; do
    prev=""
    for sd in 1 2; do
      ids=$(echo ${DEP[$lm,$DS,$sd]:-} $prev | tr ' ' ':'); d=${ids:+afterok:$ids}
      e=$(SEED=$sd sbatch --parsable ${d:+--dependency=$d} --job-name=repro_eval_${lm}_${DS}_s$sd $S/eval_embed_search.sbatch $lm $DS single $MC); prev=$e
      echo "$lm $DS seed$sd eval $e (dep $d)" >> $LOG
    done
    s=$(sbatch --parsable --dependency=afterok:$prev --job-name=repro_score_${lm}_$DS $S/eval_score.sbatch $lm $DS full)
    o=$(sbatch --parsable --dependency=afterok:$prev --job-name=repro_oldsearch_${lm}_$DS $S/search_old_index.sbatch $lm/$DS)
    so=$(sbatch --parsable --dependency=afterok:$o --job-name=repro_score_old_${lm}_$DS $S/eval_score.sbatch $lm $DS old21m)
    echo "$lm $DS score $s oldsearch $o oldscore $so" >> $LOG
  done
done
