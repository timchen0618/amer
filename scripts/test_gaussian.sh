results_dir="results/gaussian_synthetic_inf"

python test.py \
--model_paths  ${results_dir}/sm_dual_hungarian_contrastive_lr2e-5_temp0.05_batch128_ep400_warmup0.05/ \
               ${results_dir}/sm_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch64_ep100_warmup0.05/ \
               ${results_dir}/sm_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch64_ep100_warmup0.05/ \
               ${results_dir}/sm_hungarian_contrastive_lr2e-5_temp0.05_batch64_ep100_warmup0.05/ \
               ${results_dir}/sm_contrastive_first_label_lr5e-5_temp0.05_batch64_ep100_warmup0.05/ \
               ${results_dir}/sm_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch64_ep100_warmup0.05/ \
               ${results_dir}/sm_mse_all_labels_lr5e-5_temp0.05_batch64_ep100_warmup0.05/ \
               ${results_dir}/sm_mse_first_label_lr5e-5_temp0.05_batch64_ep100_warmup0.05/ \
               --checkpoint_name best_model -n 1 5  --split train --k_values 10 20 50 100 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/opposing_pairs_data/