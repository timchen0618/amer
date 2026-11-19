#!/bin/bash
#SBATCH --job-name=geom_ref
#SBATCH --time=01:30:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=16
#SBATCH --output=results/phaseC/geometry_ref/logs/%a.out
# Query/target geometry (tools/geometry_eval.py) for base INF-Retriever and the AMER reproduction's saved test
# query embeddings, on F's corpus (shared_frozen_base_512) and test file, so they compare to F's test_geom.json.
cd /scratch/hc3337/projects/autoregressive
O=results/phaseC/geometry_ref
read -r name src < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $O/list.txt)
mkdir -p $O/$name
singularity exec --overlay /scratch/hc3337/envs/nli.ext3:ro /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif bash -c "
source /ext3/env.sh; conda activate nli; cd /scratch/hc3337/projects/autoregressive
python -c \"
import numpy as np
x = np.load('$src').astype(np.float32)
x = x.reshape(531, -1, x.shape[-1])  # repro multi is stored (531*k, d)
x = x[:, 0] if x.shape[1] == 1 else x
print('$name', x.shape); np.save('$O/$name/test_qemb.npy', x)
\"
python training/inf_retriever/tools/geometry_eval.py --qemb $O/$name/test_qemb.npy \
  --emb_glob 'wikipedia_embeddings/qampari/shared_frozen_base_512/passages_*' \
  --data data/amer_data/eval_data/qampari.jsonl --out $O/$name/test_geom.json"
