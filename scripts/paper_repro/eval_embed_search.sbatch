#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=160GB
#SBATCH --account=torch_pr_152_echoi
#SBATCH --constraint=h200
#SBATCH --partition=h200_courant,h200_public
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/hc3337/projects/autoregressive/paper_repro_data/logs/eval_%x_%j.out
# Paper-repro eval steps 1-2 for one LM x dataset: query embeddings with fe0bd78 gen_ret_and_eval.py (--save_embeddings,
# same args as run_eval.sh) for every finished run, then top-500 search on the full 25.86M index.
# usage: sbatch --job-name=repro_eval_<lm>_<ds> eval_embed_search.sbatch <lm> <ds: qampari|ambiguous_qe> [cond ...]
set -euo pipefail
LM=$1; DS=$2; shift 2; CONDS=${*:-single multi_SS multi_sampling}
W=/scratch/hc3337/projects/autoregressive/.claude/worktrees/paper-repro
D=/scratch/hc3337/projects/autoregressive/paper_repro_data
R=/scratch/hc3337/projects/autoregressive
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
case $LM in llama-1b) MID=meta-llama/Llama-3.2-1B-Instruct;; llama-3b) MID=meta-llama/Llama-3.2-3B-Instruct;;
  llama-8b) MID=meta-llama/Llama-3.1-8B-Instruct;; qwen3-4b) MID=Qwen/Qwen3-4B-Instruct-2507;; esac
if [ $DS = qampari ]; then DN=qampari_5_to_8; TD=qampari; Q=$R/data/questions/qampari_dev_question_only_5_to_8_ctxs.jsonl; K=5
else DN=ambiguous_qe; TD=ambiguous_qe; Q=$R/data/questions/ambiguous_qe_dev_question_only_2_to_5_ctxs.jsonl; K=2; fi
QD=$D/eval/$LM/$DS/qemb; mkdir -p $QD
for c in $CONDS; do
  run=${LM}_${c}_0; RD=$D/results/$LM/${DS}_inf/$run
  [ -f $RD/best_model_linear.pt ] || { echo "missing $RD"; exit 1; }
  if [[ $c == single* ]]; then mnt=1; else mnt=$K; fi
  singularity exec --nv --overlay /scratch/hc3337/envs/div.ext3:ro $IMG bash -c "source /ext3/env.sh; cd $W; \
    python gen_ret_and_eval.py --data_name $DN --training_data_name $TD --split dev --retriever inf --dev_data_path $Q --use_gpu --num_shards 16 \
      --base_model_id $MID --adapter_path $RD/best_model --linear_checkpoint_path $RD/best_model_linear.pt --base_model_type $LM \
      --max_new_tokens $mnt --round_robin_percentage 1.0 --top_k_per_query 500 --top_k 500 --inference_modes all \
      --output_path $QD/$run.jsonl --save_embeddings"
  echo "embedded $run"
done
singularity exec --nv --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG bash -c "source /ext3/env.sh; conda activate nli; \
  PAPER_CODE=$W python $W/scripts/paper_repro/search.py '$R/wikipedia_embeddings/qampari/shared_frozen_base_512/*' $QD $D/eval/$LM/$DS/top500_full.pkl"
echo "searched $LM $DS"
