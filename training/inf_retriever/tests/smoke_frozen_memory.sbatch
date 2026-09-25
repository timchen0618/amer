#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=00:40:00
#SBATCH --mem=64GB
#SBATCH --job-name=smoke_frozen_mem
#SBATCH --output=sbatch_outputs/smoke_frozen_mem_%j.out


#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1


# QAMPARI variant of finetune_ambigqa_frozen_causal.sh (Phase 1 / Run A,
# full fine-tuning, no LoRA). Fills the gap noted in experiment_plan.md:
# QAMPARI previously only had a LoRA variant of Run A
# (finetune_qampari_frozen_causal_lora.sh); this is the clean full-FT
# counterpart, same hyperparameters as every other Run-A-family run this
# session (lr=1e-5, 800 steps, warmup 30, batch 150, frozen doc encoder,
# forced causal masking, full_sampling=1, loss_fn=hungarian).

# Memory smoke test: the frozen-doc-encoder template (1 GPU, 150 examples/GPU) under the
# current mixed-precision training loop. Stops after ~25 min (past the step-30 eval);
# only the logged Memory / MemReserved values matter. Separate run_name so it never
# writes into an existing experiment's checkpoint directory.

ACCEL_CONFIG=${ACCEL_CONFIG:-accelerate_config_1gpu.yaml}  # e.g. accelerate_config_2gpu_fsdp_nooffload.yaml (with --gres=gpu:2)

singularity exec --nv \
            --overlay /scratch/hc3337/envs/div.ext3:ro \
            /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh
cd /scratch/hc3337/projects/autoregressive

temperature=0.05
total_steps=800
warmup_steps=30
lr=0.00001
save_freq=30
log_freq=5
eval_freq=30
negative_hard_ratio=0.0
negative_ctxs=1
data_name=qampari

per_gpu_batch_size=150
per_gpu_eval_batch_size=150
model_name=infly
if [ \"\$model_name\" = \"infly\" ]; then
    model_path=infly/inf-retriever-v1-1.5b
else
    echo 'Invalid model name'
    exit 1
fi
training_mode=multi
loss_fn=hungarian
full_sampling=1
freeze_doc_encoder=1
force_causal=1

data_dir=/scratch/hc3337/projects/autoregressive/data/training/filtered/\${data_name}
output_dir=checkpoints/\${data_name}/
sampling_tag=\$([ "\${full_sampling}" = "1" ] && echo "fullsr" || echo "rampsr")
causal_tag=\$([ "\${force_causal}" = "1" ] && echo "causal" || echo "bidir")
run_name=smoke_frozen_mem_bs\${per_gpu_batch_size}

chunk_length=512
accumulation_steps=1
max_positive_documents=1
num_workers=2

timeout 25m accelerate launch --config_file training/inf_retriever/${ACCEL_CONFIG} --main_process_port 29508 training/inf_retriever/finetuning_multi.py \
    --train_data \$data_dir/train_data.jsonl \
    --eval_data \$data_dir/dev_data.jsonl \
    --temperature \$temperature \
    --total_steps \$total_steps \
    --warmup_steps \$warmup_steps \
    --lr \$lr \
    --save_freq \$save_freq \
    --log_freq \$log_freq \
    --eval_freq \$eval_freq \
    --negative_hard_ratio \$negative_hard_ratio \
    --negative_ctxs \$negative_ctxs \
    --per_gpu_batch_size \$per_gpu_batch_size \
    --per_gpu_eval_batch_size \$per_gpu_eval_batch_size \
    --model_path \$model_path \
    --chunk_length \$chunk_length \
    --accumulation_steps \$accumulation_steps \
    --run_name \$run_name \
    --output_dir \$output_dir \
    --training_mode \$training_mode \
    --loss_fn \$loss_fn \
    --max_positive_documents \$max_positive_documents \
    --num_workers \$num_workers \
    --norm_query \
    --norm_doc \
    --not_save \
    --freeze_doc_encoder \
    \$([ "\${force_causal}" = "1" ] && echo "--force_causal") \
    \$([ "\${full_sampling}" = "1" ] && echo "--full_sampling")
status=\$?
[ \$status -eq 124 ] && echo 'stopped by timeout as planned' && exit 0
exit \$status
"
