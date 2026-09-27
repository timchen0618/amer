#!/bin/bash
# Phase A setup: dev validation sets -> base-model top-1000 retrieval for them -> reduced
# corpora for {qampari,ambigqa} x {test,dev}. Submits three dependent SLURM jobs.
# Usage (repo root): bash training/inf_retriever/tools/phaseA_setup.sh
set -e
cd /scratch/hc3337/projects/autoregressive
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
ACC=torch_pr_152_courant
mkdir -p data/phaseA results/phaseA/base sbatch_outputs

j1=$(sbatch --parsable --account=$ACC --time=00:30:00 --mem=32G --cpus-per-task=2 --job-name=phaseA_devsets \
  --output=sbatch_outputs/phaseA_devsets.out --wrap "singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro $IMG bash -c 'source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python training/inf_retriever/tools/phaseA_make_dev_sets.py --out_dir data/phaseA'")

j2=$(sbatch --parsable --dependency=afterok:$j1 --account=$ACC --time=3:00:00 --mem=96G --cpus-per-task=8 --gres=gpu:1 --constraint=h200 \
  --job-name=phaseA_base_dev --output=sbatch_outputs/phaseA_base_dev.out --wrap "singularity exec --nv --overlay /scratch/hc3337/envs/nli.ext3:ro $IMG bash -c 'source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive; \
  for s in qampari_dev500 ambigqa_dev300; do python retrieval_base.py --model_name_or_path infly/inf-retriever-v1-1.5b --passages /scratch/hc3337/wikipedia_chunks/chunks_v5.tsv --passages_embeddings \"/scratch/hc3337/embeddings/inf/qampari_embeddings/*\" --data data/phaseA/\$s.jsonl --output_dir results/phaseA/base --projection_size 1536 --per_gpu_batch_size 4 --n_docs 1000 --num_shards 16 --use_gpu --output_file \$s.jsonl || exit 1; done'")

j3=$(sbatch --parsable --dependency=afterok:$j2 --account=$ACC --time=4:00:00 --mem=96G --cpus-per-task=2 --job-name=phaseA_corpora \
  --output=sbatch_outputs/phaseA_corpora.out --wrap "singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro $IMG bash -c 'source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python training/inf_retriever/tools/phaseA_build_reduced_corpus.py \
  --spec qampari_test:results/base_retrievers/inf/amer_data/qampari_5_to_8_ctxs.jsonl:data/amer_data/eval_data/qampari.jsonl \
  --spec ambigqa_test:results/base_retrievers/inf/amer_data/ambigqa.jsonl: \
  --spec qampari_dev500:results/phaseA/base/qampari_dev500.jsonl:data/phaseA/qampari_dev500.jsonl \
  --spec ambigqa_dev300:results/phaseA/base/ambigqa_dev300.jsonl: \
  --topk 1000 --n_random 100000 --out_dir data/phaseA/corpora'")
echo "devsets=$j1 base_dev=$j2 corpora=$j3"
