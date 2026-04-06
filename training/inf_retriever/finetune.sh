temperature=0.05
total_steps=5000
warmup_steps=200
lr=0.00001
save_freq=250
log_freq=25
eval_freq=250
negative_hard_ratio=0.0
negative_ctxs=1

per_gpu_batch_size=256
per_gpu_eval_batch_size=256
model_name="infly"
if [ "$model_name" = "infly" ]; then
    model_path="infly/inf-retriever-v1-1.5b"
else 
    echo "Invalid model name"
    exit 1
fi
training_mode="standard_org_q"

data_dir="/scratch/hc3337/projects/diverse_response/training/data/qampari_data/qampari_corpus"
output_dir="checkpoints/"
run_name="enc_trained_qampari_${model_name}_${training_mode}_finetuned_steps${total_steps}_t${temperature}_lr${lr}_ws${warmup_steps}_bs${per_gpu_batch_size}_gradchkpt"

chunk_length=512
accumulation_steps=1
max_positive_documents=1
num_workers=2

accelerate launch --main_process_port 29501 training/inf_retriever/finetuning_multi.py --train_data $data_dir/train_data.jsonl \
                                                 --eval_data $data_dir/train_eval_data.jsonl \
                                                 --temperature $temperature \
                                                 --total_steps $total_steps \
                                                 --warmup_steps $warmup_steps \
                                                 --lr $lr \
                                                 --save_freq $save_freq \
                                                 --log_freq $log_freq \
                                                 --eval_freq $eval_freq \
                                                 --negative_hard_ratio $negative_hard_ratio \
                                                 --negative_ctxs $negative_ctxs \
                                                 --per_gpu_batch_size $per_gpu_batch_size \
                                                 --per_gpu_eval_batch_size $per_gpu_eval_batch_size \
                                                 --model_path $model_path \
                                                 --chunk_length $chunk_length \
                                                 --accumulation_steps $accumulation_steps \
                                                 --run_name $run_name \
                                                 --output_dir $output_dir \
                                                 --training_mode $training_mode \
                                                 --max_positive_documents $max_positive_documents \
                                                 --num_workers $num_workers \
                                                 --norm_query \
                                                 --norm_doc