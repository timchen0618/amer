#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_152_echoi
#SBATCH --constraint=h200
#SBATCH --partition=h200_courant,h200_public
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/hc3337/projects/autoregressive/f_paper_data/logs/%x_%j.out
# F paper-stack runs: test eval of finished runs (R's eval_embed_search.sbatch + eval_score.sbatch, INF query model).
#   P1/P2 (paper code): query embeddings with gen_ret_and_eval.py from best_model (lowest dev loss)
#   P3 (F stack) and cal_* (F checkpoints scored for calibration): query embeddings from retrieval_inf.py (via QEMB_FROM=<npy> for that run, reshaped to 2-D here)
# then top-500 search on the full 25.86M index (scripts/paper_repro/search.py) and round-robin / RRF scoring with the
# paper's scorer (scripts/paper_repro/score.py), plus pairwise cosine of each question's query embeddings (qsim.py).
# usage: sbatch --job-name=f_eval_<tag> eval.sbatch <tag> <run> [<run> ...]   (run = P1_inf_multi_SS_0 etc.)
set -euo pipefail
TAG=$1; shift
W=/scratch/hc3337/projects/autoregressive/.claude/worktrees/f-paper-inf
D=/scratch/hc3337/projects/autoregressive/f_paper_data
R=/scratch/hc3337/projects/autoregressive
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
Q=$R/data/questions/qampari_dev_question_only_5_to_8_ctxs.jsonl
QD=$D/eval/$TAG/qemb; mkdir -p $QD
for run in "$@"; do
  case $run in *single*) mnt=1;; *) mnt=5;; esac
  case $run in
    P3*|cal_*) singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro $IMG bash -c "source /ext3/env.sh; python -c \"import numpy as np; a=np.load('$R/results/f_paper/$run/test_qemb.npy'); a=a.reshape(-1,a.shape[-1]); print('$run',a.shape); np.save('$QD/$run.npy',a.astype('float32'))\"";;
    *) RD=$D/results/inf/qampari_inf/$run
       [ -f $RD/best_model_linear.pt ] || { echo "missing $RD"; exit 1; }
       singularity exec --nv --overlay /scratch/hc3337/envs/div.ext3:ro $IMG bash -c "source /ext3/env.sh; cd $W; \
         python gen_ret_and_eval.py --data_name qampari_5_to_8 --training_data_name qampari --split dev --retriever inf --dev_data_path $Q --use_gpu --num_shards 16 \
           --base_model_id infly/inf-retriever-v1-1.5b --adapter_path $RD/best_model --linear_checkpoint_path $RD/best_model_linear.pt --base_model_type inf \
           --max_new_tokens $mnt --round_robin_percentage 1.0 --top_k_per_query 500 --top_k 500 --inference_modes all \
           --output_path $QD/$run.jsonl --save_embeddings";;
  esac
  echo "embedded $run"
done
singularity exec --nv --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG bash -c "source /ext3/env.sh; conda activate nli; \
  PAPER_CODE=$W python $W/scripts/paper_repro/search.py '$R/wikipedia_embeddings/qampari/shared_frozen_base_512/*' $QD $D/eval/$TAG/top500_full.pkl"
singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro $IMG bash -c "source /ext3/env.sh; \
  PAPER_CODE=$W python $W/scripts/paper_repro/score.py qampari $D/eval/$TAG/scores.json $D/eval/$TAG/top500_full.pkl; \
  python $W/scripts/f_inf/qsim.py $QD $D/eval/$TAG/qsim.json"
echo "done $TAG"
