


# echo "================================================================"
# echo "New MLP Rotation, large"
# echo "================================================================"

# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_pred_length_full_finetuning_SSPredLength_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModelSSPredLength --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_large/ --full_finetuning  --pred_length


# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_pred_length_full_finetuning_SSPredLength_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep1500_warmup0.05_srm1 --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModelSSPredLength --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_large/ --full_finetuning  --pred_length

# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_pred_length_full_finetuning_SSPredLength_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep1500_warmup0.05_srm1 --checkpoint_name best_model -n 2  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModelSSPredLength --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_large_2/ --full_finetuning  --pred_length --num_gt 2




# echo "================================================================"
# echo "New MLP Rotation, Multi-query, large"
# echo "================================================================"
# # New MLP Rotation, Multi-query, large
# # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_multi_query_inf/normalized_large_new_mlps_gaussian_rotation_multi_query_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 results/llama-1b/gaussian_new_mlps_rotation_multi_query_inf/normalized_large_new_mlps_gaussian_rotation_multi_query_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 results/llama-1b/gaussian_new_mlps_rotation_multi_query_inf/normalized_large_new_mlps_gaussian_rotation_multi_query_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_multi_query_large/ --full_finetuning 
# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_multi_query_inf/normalized_large_new_mlps_gaussian_rotation_multi_query_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_multi_query_large/ --full_finetuning 


# echo "================================================================"
# echo "New MLP Rotation, OOD, large"
# echo "================================================================"
# # New MLP Rotation, OOD, large
# # python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_ood_inf/normalized_large_new_mlps_gaussian_rotation_ood_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 results/llama-1b/gaussian_new_mlps_rotation_ood_inf/normalized_large_new_mlps_gaussian_rotation_ood_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 results/llama-1b/gaussian_new_mlps_rotation_ood_inf/normalized_large_new_mlps_gaussian_rotation_ood_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_ood_large/ --full_finetuning 
# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_ood_inf/normalized_large_new_mlps_gaussian_rotation_ood_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_ood_large/ --full_finetuning 


# echo "================================================================"
# echo "Linear, large"
# echo "================================================================"
# # New MLP Rotation, OOD, large
# python test.py --model_paths  results/llama-1b/gaussian_linear_inf/normalized_large_linear_gaussian_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 results/llama-1b/gaussian_linear_inf/normalized_large_linear_gaussian_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 results/llama-1b/gaussian_linear_inf/normalized_large_linear_gaussian_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_large/ --full_finetuning 

# python test.py --model_paths  results/llama-1b/gaussian_linear_inf/normalized_large_linear_gaussian_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_large/ --full_finetuning 



# echo "================================================================"
# echo "Linear, multi-query, large"
# echo "================================================================"
# # New MLP Rotation, OOD, large
# python test.py --model_paths  results/llama-1b/gaussian_linear_multi_query_inf/normalized_large_linear_gaussian_multi_query_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 results/llama-1b/gaussian_linear_multi_query_inf/normalized_large_linear_gaussian_multi_query_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 results/llama-1b/gaussian_linear_multi_query_inf/normalized_large_linear_gaussian_multi_query_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1   --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_multi_query_large/ --full_finetuning 

# python test.py --model_paths  results/llama-1b/gaussian_linear_multi_query_inf/normalized_large_linear_gaussian_multi_query_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_multi_query_large/ --full_finetuning 




# echo "================================================================"
# echo "Linear, ood, large"
# echo "================================================================"
# # New MLP Rotation, OOD, large
# python test.py --model_paths  results/llama-1b/gaussian_linear_ood_inf/normalized_large_linear_gaussian_ood_full_finetuning_SS_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 results/llama-1b/gaussian_linear_ood_inf/normalized_large_linear_gaussian_ood_full_finetuning_SS_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 results/llama-1b/gaussian_linear_ood_inf/normalized_large_linear_gaussian_ood_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1   --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_ood_large/ --full_finetuning 

# python test.py --model_paths  results/llama-1b/gaussian_linear_ood_inf/normalized_large_linear_gaussian_ood_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_ood_large/ --full_finetuning 



# ####### Similarity Analysis #######
# # Linear, large
# python test.py --model_paths  results/llama-1b/gaussian_linear_inf/normalized_large_linear_gaussian_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_large/ --full_finetuning 
# # Linear, multi-query, large
# python test.py --model_paths results/llama-1b/gaussian_linear_multi_query_inf/normalized_large_linear_gaussian_multi_query_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1   --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_multi_query_large/ --full_finetuning 
# # Linear, ood, large
# python test.py --model_paths results/llama-1b/gaussian_linear_ood_inf/normalized_large_linear_gaussian_ood_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1   --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_ood_large/ --full_finetuning 
# # New MLP, large
# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_pred_length_full_finetuning_SSPredLength_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModelSSPredLength --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_large/ --full_finetuning  --pred_length
# # New MLP Rotation, Multi-query, large
# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_multi_query_inf/normalized_large_new_mlps_gaussian_rotation_multi_query_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_multi_query_large/ --full_finetuning 
# # New MLP Rotation, OOD, large
# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_ood_inf/normalized_large_new_mlps_gaussian_rotation_ood_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_ood_large/ --full_finetuning 





######## New Evaluation #####

echo "================================================================"
echo "New MLP Rotation, large"
echo "================================================================"

# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05/ --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_large/ --full_finetuning 

python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_inf/normalized_large_new_mlps_gaussian_rotation_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05 --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_large/ --full_finetuning 


echo "================================================================"
echo "New MLP Rotation, Multi-query, large"
echo "================================================================"
# New MLP Rotation, Multi-query, large
# python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_multi_query_inf/normalized_large_new_mlps_gaussian_rotation_multi_query_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_multi_query_large/ --full_finetuning 

python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_multi_query_inf/normalized_large_new_mlps_gaussian_rotation_multi_query_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_multi_query_large/ --full_finetuning 


echo "================================================================"
echo "New MLP Rotation, OOD, large"
echo "================================================================"
# New MLP Rotation, OOD, large
# python test.py --model_paths results/llama-1b/gaussian_new_mlps_rotation_ood_inf/normalized_large_new_mlps_gaussian_rotation_ood_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_ood_large/ --full_finetuning 

python test.py --model_paths  results/llama-1b/gaussian_new_mlps_rotation_ood_inf/normalized_large_new_mlps_gaussian_rotation_ood_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1 --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/new_mlps_rotation_ood_large/ --full_finetuning 


echo "================================================================"
echo "Linear, large"
echo "================================================================"
# New MLP Rotation, OOD, large
# python test.py --model_paths  results/llama-1b/gaussian_linear_inf/normalized_large_linear_gaussian_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_large/ --full_finetuning 

python test.py --model_paths  results/llama-1b/gaussian_linear_inf/normalized_large_linear_gaussian_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_large/ --full_finetuning 



echo "================================================================"
echo "Linear, multi-query, large"
echo "================================================================"
# New MLP Rotation, OOD, large
# python test.py --model_paths  results/llama-1b/gaussian_linear_multi_query_inf/normalized_large_linear_gaussian_multi_query_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1   --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_multi_query_large/ --full_finetuning 

python test.py --model_paths  results/llama-1b/gaussian_linear_multi_query_inf/normalized_large_linear_gaussian_multi_query_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_multi_query_large/ --full_finetuning 




echo "================================================================"
echo "Linear, ood, large"
echo "================================================================"
# New MLP Rotation, OOD, large
# python test.py --model_paths  results/llama-1b/gaussian_linear_ood_inf/normalized_large_linear_gaussian_ood_full_finetuning_SS_hungarian_contrastive_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1   --checkpoint_name best_model -n 5  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_ood_large/ --full_finetuning 

python test.py --model_paths  results/llama-1b/gaussian_linear_ood_inf/normalized_large_linear_gaussian_ood_full_finetuning_SS_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch128_ep3000_warmup0.05_srm1  --checkpoint_name best_model -n 1  --split test --k_values 1 5 10 20 50 100 500 --model_type EmbeddingModel --embedding_model_dim 1024 -d data_creation/gaussian/data/linear_ood_large/ --full_finetuning 
