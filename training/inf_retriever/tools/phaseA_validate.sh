#!/bin/bash
# Phase A: cheap-eval every checkpoint in phaseA_validation_manifest.tsv on the reduced
# corpora. SPLIT=test (default) validates against the known full-corpus test scores;
# SPLIT=dev gives the dev-query reference scores that phase C selection compares against.
# Usage (repo root): [SPLIT=dev] bash training/inf_retriever/tools/phaseA_validate.sh [dependency-jobid]
cd /scratch/hc3337/projects/autoregressive
DEP=""; [ -n "${1:-}" ] && DEP="--dependency=afterok:$1"
SPLIT=${SPLIT:-test}
# '|' instead of a tab as IFS: bash collapses runs of whitespace IFS characters, which
# would drop the empty k field of single-query rows.
tail -n +2 training/inf_retriever/tools/phaseA_validation_manifest.tsv | tr '\t' '|' | while IFS='|' read -r ds name ckpt k full_mr full_r; do
  if [ "$SPLIT" = test ]; then set=${ds}_test; data=data/amer_data/eval_data/${ds}.jsonl; out=results/phaseA/validation/$ds/$name
  else set=$( [ $ds = qampari ] && echo qampari_dev500 || echo ambigqa_dev300 ); data=data/phaseA/$set.jsonl; out=results/phaseA/dev/$ds/$name; fi
  jn=phaseA_${SPLIT}_${ds}_${name}
  if grep -q MRecall $out/eval_metrics.txt 2>/dev/null; then echo "skip $ds/$name (done)"; continue; fi
  if squeue -h -u $USER -n $jn | grep -q .; then echo "skip $ds/$name (already queued)"; continue; fi
  j=$(sbatch --parsable $DEP --job-name=$jn \
      --export=ALL,CKPT=$ckpt,SET=$set,DATA=$data,OUT_DIR=$out,K=$k \
      training/inf_retriever/tools/cheap_eval_ckpt.sbatch)
  echo "$ds/$name -> job $j"
done
