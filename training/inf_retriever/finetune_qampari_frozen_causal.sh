#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --tasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=qampari_infly_frozen_causal_multi_hungarian
#SBATCH --output=sbatch_outputs/qampari_infly_frozen_causal_multi_hungarian.out
#SBATCH --mail-type=END
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1

# NOTE (2026-09-25, branch fsdp-clean-recipe): under the current code this script
# trains in MIXED precision (fp32 weights + fp32 AdamW, bf16 autocast) with a
# correctly-stepped LR scheduler and seeded, rank-consistent data order. It is a
# TEMPLATE, not the fallback recipe. The fallback (pure bf16, 2x-compressed LR
# schedule, unseeded shuffling) is only reproducible from git tag
# fallback-bf16-ddp -- see RECIPE_FALLBACK_bf16_ddp.md.

# QAMPARI variant of finetune_ambigqa_frozen_causal.sh (Phase 1 / Run A,
# full fine-tuning, no LoRA). Fills the gap noted in experiment_plan.md:
# QAMPARI previously only had a LoRA variant of Run A
# (finetune_qampari_frozen_causal_lora.sh); this is the clean full-FT
# counterpart, same hyperparameters as every other Run-A-family run this
# session (lr=1e-5, 800 steps, warmup 30, batch 150, frozen doc encoder,
# forced causal masking, full_sampling=1, loss_fn=hungarian).

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
run_name=frozenDocEnc_\${causal_tag}_\${data_name}_\${model_name}_\${training_mode}_finetuned_steps\${total_steps}_t\${temperature}_lr\${lr}_ws\${warmup_steps}_bs\${per_gpu_batch_size}_\${loss_fn}_\${sampling_tag}

chunk_length=512
accumulation_steps=1
max_positive_documents=1
num_workers=2

accelerate launch --config_file training/inf_retriever/accelerate_config_1gpu.yaml --main_process_port 29508 training/inf_retriever/finetuning_multi.py \
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
    --freeze_doc_encoder \
    \$([ "\${force_causal}" = "1" ] && echo "--force_causal") \
    \$([ "\${full_sampling}" = "1" ] && echo "--full_sampling")
"
