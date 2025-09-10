# results_dir="results/gaussian_diverse_mlps_sample_transformation_inf"

# python test.py \
# --model_paths  ${results_dir}/gaussian_full_finetuning_mse_all_labels_lr2e-5_temp0.05_batch128_ep500_warmup0.05/ ${results_dir}/gaussian_full_finetuning_hungarian_contrastive_lr2e-5_temp0.05_batch128_ep500_warmup0.05/ ${results_dir}/gaussian_full_finetuning_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch128_ep500_warmup0.05/ --checkpoint_name best_model -n 1 5  --split train --k_values 10 20 50 100 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/opposing_pairs_data_large/ --full_finetuning 


# python test.py \
# --model_paths  ${results_dir}/sm_full_finetuning_SS_fixed_hungarian_contrastive_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ ${results_dir}/sm_full_finetuning_SS_fixed_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ ${results_dir}/sm_full_finetuning_SS_fixed_mse_all_labels_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ ${results_dir}/sm_full_finetuning_SSVariable_hungarian_contrastive_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ --checkpoint_name best_model -n 1 5  --split train --k_values 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/opposing_pairs_data/ --full_finetuning 

# echo "================================================================"
# echo "MLP, small"
# echo "================================================================"
# # MLP, small
# python test.py --model_paths results/gaussian_diverse_mlps_inf/normalized_mlps_gaussian_full_finetuning_SS_contrastive_first_label_lr5e-5_temp0.05_batch32_ep32000_warmup0.05/ --checkpoint_name best_model -n 1 5  --split train --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModelSS --embedding_model_dim 1024 -d data_creation/gaussian/data/diverse_mlps_data/ --full_finetuning 


# echo "================================================================"
# echo "MLP, large"
# echo "================================================================"
# # MLP, large
# python test.py --model_paths  results/llama-1b/gaussian_diverse_mlps_inf/normalized_large_mlps_gaussian_4gpu_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/  --checkpoint_name best_model -n 1 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModelSS --embedding_model_dim 1024 -d data_creation/gaussian/data/diverse_mlps_large/ --full_finetuning 



# echo "================================================================"
# echo "Linear, multi-query, small"
# echo "================================================================"
# # Linear, multi-query, small
# python test.py --model_paths  results/gaussian_linear_multi_query_inf/linear_gaussian_multi_query_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep32000_warmup0.05 results/gaussian_linear_multi_query_inf/linear_gaussian_multi_query_full_finetuning_SS_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep16000_warmup0.05 results/gaussian_linear_multi_query_inf/linear_gaussian_multi_query_full_finetuning_SS_contrastive_first_label_lr2e-5_temp0.05_batch32_ep400_warmup0.05  --checkpoint_name best_model -n 1 5  --split train --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_multi_query/ --full_finetuning 

# echo "================================================================"
# echo "Linear, multi-query, large"
# echo "================================================================"
# # Linear, multi-query, large
# python test.py --model_paths  results/llama-1b/gaussian_linear_multi_query_inf/normalized_large_linear_gaussian_multi_query_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/  --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_multi_query_large/ --full_finetuning 

# echo "================================================================"
# echo "MLP, multi-query, small"
# echo "================================================================"
# # MLP, multi-query, small
# python test.py --model_paths  results/gaussian_diverse_mlps_multi_query_inf/normalized_mlps_gaussian_multi_query_full_finetuning_SS_contrastive_first_label_lr5e-5_temp0.05_batch32_ep32000_warmup0.05/ --checkpoint_name best_model -n 1   --split train --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/diverse_mlps_multi_query/ --full_finetuning 

# echo "================================================================"
# echo "MLP, multi-query, large"
# echo "================================================================"
# # MLP, multi-query, large
# python test.py --model_paths results/llama-1b/gaussian_diverse_mlps_multi_query_inf/normalized_large_mlps_gaussian_multi_query_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/  --checkpoint_name best_model -n 1 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/diverse_mlps_multi_query_large/ --full_finetuning





# echo "================================================================"
# echo "MLP, ood, large"
# echo "================================================================"
# python test.py --model_paths results/llama-1b/gaussian_diverse_mlps_ood_inf/normalized_large_mlps_gaussian_ood_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/ results/llama-1b/gaussian_diverse_mlps_ood_inf/normalized_large_mlps_gaussian_ood_full_finetuning_SS_contrastive_first_label_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/  --checkpoint_name best_model -n 1 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/diverse_mlps_ood_large/ --full_finetuning


# echo "================================================================"
# echo "New MLP, large"
# echo "================================================================"
# # New MLP, large
# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_inf/normalized_large_new_mlps_gaussian_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_data_large/ --full_finetuning

# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_inf/normalized_large_new_mlps_gaussian_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/ results/llama-1b/gaussian_new_mlps_inf/normalized_large_new_mlps_gaussian_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/ results/llama-1b/gaussian_new_mlps_inf/normalized_large_new_mlps_gaussian_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/ --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_data_large/ --full_finetuning 



# echo "================================================================"
# echo "New MLP Harder, large"
# echo "================================================================"
# # # New MLP, large
# # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_harder_inf/normalized_large_new_mlps_gaussian_harder_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_harder_data_large/ --full_finetuning 

# # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_harder_inf/normalized_large_new_mlps_gaussian_harder_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 results/llama-1b/gaussian_new_mlps_harder_inf/normalized_large_new_mlps_gaussian_harder_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 results/llama-1b/gaussian_new_mlps_harder_inf/normalized_large_new_mlps_gaussian_harder_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_harder_data_large/ --full_finetuning 

# python test.py --model_paths results/llama-1b/gaussian_new_mlps_harder_inf/normalized_large_new_mlps_gaussian_harder_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05  results/llama-1b/gaussian_new_mlps_harder_inf/normalized_large_new_mlps_gaussian_harder_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 --checkpoint_name best_model -n 1 3  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_harder_data_large/ --full_finetuning 

# echo "================================================================"
# echo "New MLP Rotation, large"
# echo "================================================================"
# # # # New MLP, large
# # # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_large/ --full_finetuning 

# # # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_large/ --full_finetuning 


# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_inf/less_ss_normalized_large_new_mlps_gaussian_rotation_full_finetuning_SS_contrastive_all_labels_shuffled_woseq_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/ --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_large/ --full_finetuning 

# echo "================================================================"
# echo "New MLP Normal, large"
# echo "================================================================"
# # # # New MLP, large
# # # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_normal_inf/normalized_large_new_mlps_gaussian_normal_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_normal_large/ --full_finetuning 

# # # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_normal_inf/normalized_large_new_mlps_gaussian_normal_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 results/llama-1b/gaussian_new_mlps_normal_inf/normalized_large_new_mlps_gaussian_normal_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 results/llama-1b/gaussian_new_mlps_normal_inf/normalized_large_new_mlps_gaussian_normal_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 results/llama-1b/gaussian_new_mlps_normal_inf/less_ss_normalized_large_new_mlps_gaussian_normal_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/  --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_normal_large/ --full_finetuning 

# # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_normal_inf/normalized_large_new_mlps_gaussian_normal_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05   results/llama-1b/gaussian_new_mlps_normal_inf/less_ss_normalized_large_new_mlps_gaussian_normal_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/  --checkpoint_name best_model -n 1 3  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_normal_large/ --full_finetuning 


# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_normal_inf/less_ss_normalized_large_new_mlps_gaussian_normal_full_finetuning_SS_contrastive_all_labels_shuffled_woseq_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/ --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_normal_large/ --full_finetuning

# # # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_normal_inf/normalized_large_new_mlps_gaussian_normal_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 results/llama-1b/gaussian_new_mlps_normal_inf/less_ss_normalized_large_new_mlps_gaussian_normal_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/ --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_normal_large/ --full_finetuning --use_ground_truth_for_eval


# echo "================================================================"
# echo "New MLP Opposite, large"
# echo "================================================================"
# # # # New MLP, large
# # # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_opposite_inf/less_ss_normalized_large_new_mlps_gaussian_opposite_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_opposite_large/ --full_finetuning 

# # # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_opposite_inf/less_ss_normalized_large_new_mlps_gaussian_opposite_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 results/llama-1b/gaussian_new_mlps_opposite_inf/less_ss_normalized_large_new_mlps_gaussian_opposite_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 results/llama-1b/gaussian_new_mlps_opposite_inf/less_ss_normalized_large_new_mlps_gaussian_opposite_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/ --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_opposite_large/ --full_finetuning 

# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_opposite_inf/less_ss_normalized_large_new_mlps_gaussian_opposite_full_finetuning_SS_contrastive_all_labels_shuffled_woseq_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_opposite_large/ --full_finetuning 





echo "================================================================"
echo "New MLP Rotation, large"
echo "================================================================"

# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_pred_length_full_finetuning_SSPredLength_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModelSSPredLength --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_large/ --full_finetuning  --pred_length


python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_pred_length_full_finetuning_SSPredLength_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep1500_warmup0.05_srm1 --checkpoint_name best_model -n 2  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModelSSPredLength --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_large_2/ --full_finetuning  --pred_length