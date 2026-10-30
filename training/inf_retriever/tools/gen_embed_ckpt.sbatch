#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=96GB
#SBATCH --account=torch_pr_152_echoi
#SBATCH --constraint=h200|l40s|b200
#SBATCH --partition=h200_courant,l40s_courant,h200_public,l40s_public,h200,b200
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --gres=gpu:1
#SBATCH --array=0-31
#SBATCH --job-name=gen_embed_ckpt
#SBATCH --output=sbatch_outputs/gen_embed_%x_%a.out

# 2026-10-10: --mem 64GB -> 96GB: shard jobs peaked at 63 GB host RAM on L40S and 4 were OOM-killed.
# 2026-10-08: b200 back (dropped 2026-10-07). 345 of 352 B200 tasks since 2026-09-29 succeeded; the 7 failures were
# "CUDA-capable device(s) is/are busy or unavailable" at start-up, on nodes (gb006, gb011, gb012) that also ran 18-40
# tasks fine. Such a start-up failure now requeues the task (at most twice) instead of failing it.
# GPU: H200, L40S or B200 (2026-09-29: widens the pool under low priority). Only the b200 and h200 partitions
# in the list above are preemptible (the others are the group's own/public non-preemptible ones);
# a preempted task is requeued and simply re-embeds its shard. Preemptible L40S is left out (a shard
# takes ~74 min there; jobs are protected only for their first 30 minutes). Preemptible H200 was added
# 2026-10-01: a shard takes ~29 min (98% finish within 30 min); B200 takes ~19 min.
# The div environment's PyTorch (2.5.1+cu121) has no B200 (sm_100) kernels, so on B200 the nli
# environment (PyTorch 2.8.0+cu128) is used instead: on 2,000 passages the two agree to cosine
# 0.999983 at worst (mean 0.999998), with 100% identical nearest neighbours.
# Batch size by GPU memory: 512 on H200 (141 GB, as before); 128 on smaller GPUs, since a batch of
# 512 long passages ran out of memory on an L40S (44 GB). Per-passage embeddings do not depend on
# the batch (right padding + attention mask), up to floating-point noise.
# Passage truncation (code audit 2026-09-28, R3): 512 tokens = training's --chunk_length. The 24-run
# grid's checkpoints (grid_*) were all evaluated at 1024, the old hard-coded value, and keep it so
# their results stay comparable. PASSAGE_MAXLEN overrides both.
case "$CKPT" in */grid_*|grid_*) PML_DEFAULT=1024 ;; *) PML_DEFAULT=512 ;; esac
PML=${PASSAGE_MAXLEN:-$PML_DEFAULT}
echo "passage max length $PML tokens"
GPU_MEM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
BS=128; [ -n "$GPU_MEM_MB" ] && [ "$GPU_MEM_MB" -ge 80000 ] && BS=512
echo "GPU memory ${GPU_MEM_MB:-unknown} MiB -> per_gpu_batch_size $BS"
# Embed the full passage corpus with one joint-trained checkpoint (32-way array,
# one ~808K-passage shard per task, ~75 GB total). Parameterized version of the
# per-checkpoint gen_embed_*.sbatch scripts.
#   sbatch --job-name=<tag> --export=ALL,CKPT=<checkpoint dir>,EMB_DIR=<output dir> \
#       training/inf_retriever/tools/gen_embed_ckpt.sbatch
: "${CKPT:?set CKPT}" "${EMB_DIR:?set EMB_DIR}"
PASSAGES="/scratch/hc3337/wikipedia_chunks/chunks_v5.tsv"
mkdir -p "$EMB_DIR"

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
OVERLAY=/scratch/hc3337/envs/div.ext3; ACTIVATE=""
case "$GPU_NAME" in *B200*) OVERLAY=/scratch/hc3337/envs/nli.ext3; ACTIVATE="conda activate nli;" ;; esac
echo "GPU $GPU_NAME -> environment $OVERLAY"

ELOG=sbatch_outputs/gen_embed_${SLURM_JOB_NAME}_${SLURM_ARRAY_TASK_ID}.busycheck.log
singularity exec --nv \
            --overlay $OVERLAY:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "source /ext3/env.sh; $ACTIVATE cd /scratch/hc3337/projects/autoregressive; python gen_embed_new.py --passages $PASSAGES --output_dir $EMB_DIR --shard_id ${SLURM_ARRAY_TASK_ID} --num_shards 32 --model_name_or_path $CKPT --per_gpu_batch_size $BS --passage_maxlength $PML" 2>&1 | tee $ELOG
rc=${PIPESTATUS[0]}
if [ $rc -ne 0 ] && grep -q "busy or unavailable" $ELOG && [ "${SLURM_RESTART_COUNT:-0}" -lt 2 ]; then
  echo "CUDA device busy on $(hostname); requeueing (restart ${SLURM_RESTART_COUNT:-0})"; rm -f $ELOG
  scontrol requeue $SLURM_JOB_ID; sleep 60
fi
rm -f $ELOG
exit $rc
