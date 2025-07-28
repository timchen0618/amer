results_dir="results/gaussian_diverse_mlps_inf"

# python test.py \
# --model_paths  ${results_dir}/gaussian_full_finetuning_mse_all_labels_lr2e-5_temp0.05_batch128_ep500_warmup0.05/ ${results_dir}/gaussian_full_finetuning_hungarian_contrastive_lr2e-5_temp0.05_batch128_ep500_warmup0.05/ ${results_dir}/gaussian_full_finetuning_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch128_ep500_warmup0.05/ --checkpoint_name best_model -n 1 5  --split train --k_values 10 20 50 100 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/opposing_pairs_data_large/ --full_finetuning 


# python test.py \
# --model_paths  ${results_dir}/sm_full_finetuning_SS_fixed_hungarian_contrastive_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ ${results_dir}/sm_full_finetuning_SS_fixed_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ ${results_dir}/sm_full_finetuning_SS_fixed_mse_all_labels_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ ${results_dir}/sm_full_finetuning_SSVariable_hungarian_contrastive_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ --checkpoint_name best_model -n 1 5  --split train --k_values 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/opposing_pairs_data/ --full_finetuning 


python test.py \
--model_paths  ${results_dir}/mlps_gaussian_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch512_ep1000_warmup0.05/ ${results_dir}/mlps_gaussian_full_finetuning_SS_mse_all_labels_lr5e-5_temp0.05_batch512_ep1000_warmup0.05/ ${results_dir}/mlps_gaussian_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch512_ep1000_warmup0.05/ ${results_dir}/mlps_gaussian_full_finetuning_SS_contrastive_first_label_lr5e-5_temp0.05_batch512_ep1000_warmup0.05/ --checkpoint_name best_model -n 1 5  --split train --k_values 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/diverse_mlps_data/ --full_finetuning 


