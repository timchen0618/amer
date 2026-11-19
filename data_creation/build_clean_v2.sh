#!/bin/bash
# Rebuild the v2 QAMPARI data end to end (data/training/clean_v2/README.md documents every step).
# Submits four dependent SLURM jobs:
#   1. build_clean_v2.py           CPU   splits -> data/training/clean_v2/qampari/
#   2. phaseA_make_dev_sets.py     CPU   dev eval files -> data/phaseA/<ds>_cleanv2dev500.jsonl
#   3. clean_v2_base_retrieval     GPU   base top-1000 (dev) and top-100 (train)
#   4. clean_v2_downstream         CPU   hard negatives, reduced corpus, AmbigQA links, verification
# Usage (repo root): bash data_creation/build_clean_v2.sh
set -e
cd /scratch/hc3337/projects/autoregressive
IMG=/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif
ACC=torch_pr_152_courant
RUN="singularity exec --overlay /scratch/hc3337/envs/div.ext3:ro $IMG bash -c"
mkdir -p sbatch_outputs data/training/clean_v2

j1=$(sbatch --parsable --account=$ACC --time=03:00:00 --mem=96G --cpus-per-task=2 --job-name=build_clean_v2 \
  --output=sbatch_outputs/build_clean_v2.out \
  --wrap "$RUN 'source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python data_creation/build_clean_v2.py && ln -sfn ../clean/ambigqa data/training/clean_v2/ambigqa'")
j2=$(sbatch --parsable --dependency=afterok:$j1 --account=$ACC --time=00:30:00 --mem=32G --cpus-per-task=2 --job-name=v2_devsets \
  --output=sbatch_outputs/v2_devsets.out \
  --wrap "$RUN 'source /ext3/env.sh; cd /scratch/hc3337/projects/autoregressive; python training/inf_retriever/tools/phaseA_make_dev_sets.py --src_root data/training/clean_v2 --tag cleanv2 --out_dir data/phaseA'")
j3=$(sbatch --parsable --dependency=afterok:$j2 data_creation/clean_v2_base_retrieval.sbatch)
j4=$(sbatch --parsable --dependency=afterok:$j3 data_creation/clean_v2_downstream.sbatch)
echo "build=$j1 devsets=$j2 base_retrieval=$j3 downstream=$j4"
echo "When $j4 finishes, sbatch_outputs/clean_v2_downstream.out must end with 'ALL CHECKS PASSED' and 'ALL_DONE'."
