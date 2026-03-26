# results_dir="results/gaussian_diverse_mlps_sample_transformation_inf"

# python test.py \
# --model_paths  ${results_dir}/gaussian_full_finetuning_mse_all_labels_lr2e-5_temp0.05_batch128_ep500_warmup0.05/ ${results_dir}/gaussian_full_finetuning_hungarian_contrastive_lr2e-5_temp0.05_batch128_ep500_warmup0.05/ ${results_dir}/gaussian_full_finetuning_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch128_ep500_warmup0.05/ --checkpoint_name best_model -n 1 5  --split train --k_values 10 20 50 100 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/opposing_pairs_data_large/ --full_finetuning 


# python test.py \
# --model_paths  ${results_dir}/sm_full_finetuning_SS_fixed_hungarian_contrastive_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ ${results_dir}/sm_full_finetuning_SS_fixed_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ ${results_dir}/sm_full_finetuning_SS_fixed_mse_all_labels_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ ${results_dir}/sm_full_finetuning_SSVariable_hungarian_contrastive_lr2e-5_temp0.05_batch16_ep100_warmup0.05/ --checkpoint_name best_model -n 1 5  --split train --k_values 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/opposing_pairs_data/ --full_finetuning 

echo "================================================================"
echo "MLP, small"
echo "================================================================"
# MLP, small
python test.py --model_paths results/gaussian_diverse_mlps_inf/normalized_mlps_gaussian_full_finetuning_SS_contrastive_first_label_lr5e-5_temp0.05_batch32_ep32000_warmup0.05/ --checkpoint_name best_model -n 1 5  --split train --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModelSS --embedding_model_dim 1024 -d data_creation/gaussian/data/diverse_mlps_data/ --full_finetuning 


echo "================================================================"
echo "MLP, large"
echo "================================================================"
# MLP, large
python test.py --model_paths  results/gaussian_diverse_mlps_inf/normalized_large_mlps_gaussian_full_finetuning_SS_contrastive_first_label_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/  --checkpoint_name best_model -n 1 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModelSS --embedding_model_dim 1024 -d data_creation/gaussian/data/diverse_mlps_large/ --full_finetuning 



# echo "================================================================"
# echo "Linear, multi-query, small"
# echo "================================================================"
# # Linear, multi-query, small
# python test.py --model_paths  results/gaussian_linear_multi_query_inf/linear_gaussian_multi_query_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep32000_warmup0.05 results/gaussian_linear_multi_query_inf/linear_gaussian_multi_query_full_finetuning_SS_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep16000_warmup0.05 results/gaussian_linear_multi_query_inf/linear_gaussian_multi_query_full_finetuning_SS_contrastive_first_label_lr2e-5_temp0.05_batch32_ep400_warmup0.05  --checkpoint_name best_model -n 1 5  --split train --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_multi_query/ --full_finetuning 

echo "================================================================"
echo "Linear, multi-query, large"
echo "================================================================"
# Linear, multi-query, large
python test.py --model_paths  results/gaussian_linear_multi_query_inf/large_linear_gaussian_multi_query_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch512_ep4000_warmup0.05/  --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_multi_query_large/ --full_finetuning 

echo "================================================================"
echo "MLP, multi-query, small"
echo "================================================================"
# MLP, multi-query, small
python test.py --model_paths  results/gaussian_diverse_mlps_multi_query_inf/normalized_mlps_gaussian_multi_query_full_finetuning_SS_contrastive_first_label_lr5e-5_temp0.05_batch32_ep32000_warmup0.05/ --checkpoint_name best_model -n 1   --split train --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/diverse_mlps_multi_query/ --full_finetuning 

echo "================================================================"
echo "MLP, multi-query, large"
echo "================================================================"
# MLP, multi-query, large
python test.py --model_paths results/gaussian_diverse_mlps_multi_query_inf/normalized_large_mlps_gaussian_multi_query_full_finetuning_SS_contrastive_first_label_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/  --checkpoint_name best_model -n 1   --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/diverse_mlps_multi_query_large/ --full_finetuning





# echo "================================================================"
# echo "MLP, ood, large"
# echo "================================================================"
# python test.py --model_paths results/gaussian_diverse_mlps_ood_inf/normalized_large_mlps_gaussian_ood_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05  results/gaussian_diverse_mlps_ood_inf/normalized_large_mlps_gaussian_ood_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05  --checkpoint_name best_model -n 1 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/diverse_mlps_ood_large/ --full_finetuning
