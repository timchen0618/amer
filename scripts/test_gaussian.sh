results_dir="results/gaussian_synthetic_inf"

python test.py \
--model_paths  ${results_dir}/sm_full_finetuning_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch128_ep500_warmup0.05/ \
               --checkpoint_name best_model -n 1 5  --split train --k_values 10 20 50 100 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/opposing_pairs_data/ --full_finetuning --use_ground_truth_for_eval