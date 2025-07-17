from pathlib import Path
import numpy as np
from test import load_synthetic_dataset, eval_metrics, eval_on_each_gt
from data_creation.gaussian.eval_utils import compute_recall_at_k, compute_mrecall_at_k
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def compute_percentage_of_randomness(CUTOFF, rankings, topks):
    percentages = []
    for topk in topks:
        rankings_topk = rankings[:, :topk]
        num_random = np.sum(rankings_topk >= CUTOFF)
        percentage_random = num_random / (rankings_topk.shape[0] * topk)
        print(f'Top {topk}: {percentage_random}')
        percentages.append(percentage_random)
    return percentages

def compute_cosine_similarity(query, doc):
    return np.dot(query, doc) / (np.linalg.norm(query) * np.linalg.norm(doc))

def compute_prediction_similarity(rankings, corpus, test_pairs, topks):
    assert len(rankings) == len(test_pairs)
    similarities = []
    for topk in topks:
        topk_similarities = []
        for i in range(len(test_pairs)):
            pair = test_pairs[i]
            gts = corpus[np.array(pair['ground_truth_indices'])]
            # print('gts', gts.shape)
            rankings_topk = corpus[rankings[i, :topk]]
            # print('rankings_topk', rankings_topk.shape)
            cos_sims = np.max(cosine_similarity(rankings_topk, gts), axis=1)
            # print('cos_sims', cos_sims.shape)
            topk_similarities.append(np.mean(cos_sims))
            
        similarities.append(sum(topk_similarities) / len(topk_similarities))
    return similarities


# def eval_metrics(rankings, test_pairs, k_values, _print=True):
#     results = {}
#     for k in k_values:
#         recall = compute_recall_at_k(rankings, test_pairs, k)
#         mrecall = compute_mrecall_at_k(rankings, test_pairs, k)
        
#         results[f'recall@{k}'] = recall
#         results[f'mrecall@{k}'] = mrecall
#         if _print:
#             print(f"  Recall@{k}: {recall:.4f}")
#             print(f"  MRecall@{k}: {mrecall:.4f}")
#     return results

if __name__ == '__main__':
    folder_list = ['sm_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch64_ep100_warmup0.05/',
               'sm_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch64_ep100_warmup0.05/',
               'sm_hungarian_contrastive_lr2e-5_temp0.05_batch64_ep100_warmup0.05/', 
               'sm_contrastive_first_label_lr5e-5_temp0.05_batch64_ep100_warmup0.05/',
               'sm_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch64_ep100_warmup0.05/',
               'sm_mse_all_labels_lr5e-5_temp0.05_batch64_ep100_warmup0.05/',
               'sm_mse_first_label_lr5e-5_temp0.05_batch64_ep100_warmup0.05/',
               ]

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', '-c', type=str, default='randomness')
    parser.add_argument('--rootdir', '-r', type=str, default='results/gaussian_synthetic_inf/')
    parser.add_argument('--data_dir', '-d', type=str, default='data_creation/gaussian/data/opposing_pairs_data/')
    parser.add_argument('--split', '-s', type=str, default='train')
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    test_pairs, queries, corpus = load_synthetic_dataset(data_dir=str(DATA_DIR), split=args.split)
    
    if args.command == 'randomness':
        CUTOFF=11000
        max_new_tokens = 1
        topk = 5
        results = []
        for folder in folder_list:
            model_path = Path(args.rootdir) / folder
            _id = '_'.join(folder.split('lr')[0].split('_')[1:-1])
            if folder == 'random_baseline/':
                rankings = np.load(model_path / f'random_baseline_rankings.npy')
            else:
                rankings = np.load(model_path / f'max_new_tokens_{max_new_tokens}_rankings.npy')
            topks = [1, 5, 10, 20, 50, 100]
            print(f'{_id}:')
            percentages = compute_percentage_of_randomness(CUTOFF, rankings, topks)
            results.append([_id] + percentages)

        import pandas as pd
        results = pd.DataFrame(results, columns=['model'] + [f'{k}' for k in topks])
        results.to_csv('results/gaussian_synthetic_inf/randomness.csv', index=False)
        
    elif args.command == 'query_similarity':
        """
            Whether the queries are similar to each other. 
            Compute the minimum cosine similarity for each query with all other queries. 
        """
        results = [] # len(queries)
        # compute the minimum cosine similarity for each query with all other queries. 
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(queries, queries)
        for i in range(len(queries)):
            similarities[i, i] = 0
        for i in range(len(queries)):
            max_similarity = np.max(similarities[i, :])
            results.append(max_similarity)
        results = np.array(results)
        print('max, min, mean, median:', np.max(results), np.min(results), np.mean(results), np.median(results))
        # print(list(results))
    elif args.command == 'prediction_similarity':
        """
            Whether the predictions are similar to the ground truth. 
            Divided by the top k. 
        """
        folder_list = ['random_baseline/']
        max_new_tokens = 1
        results = []
        for folder in folder_list:
            model_path = Path(args.rootdir) / folder
            _id = '_'.join(folder.split('lr')[0].split('_')[1:-1])
            if folder == 'random_baseline/':
                rankings = np.load(model_path / f'random_baseline_rankings.npy')
            else:
                rankings = np.load(model_path / f'max_new_tokens_{max_new_tokens}_rankings.npy')
            topks = [1, 5, 10, 20, 50, 100]
            print(f'{_id}:')
            similarity = compute_prediction_similarity(rankings, corpus, test_pairs, topks)
            results.append([_id] + similarity)
            print(similarity)

        import pandas as pd
        results = pd.DataFrame(results, columns=['model'] + [f'{k}' for k in topks])
        results.to_csv('results/gaussian_synthetic_inf/prediction_similarity.csv', index=False)
    
    elif args.command == 'see_predictions':
        ### Initialization
        max_new_tokens = 5
        TOPK = 10
        results = []
        folder = 'sm_full_finetuning_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch128_ep500_warmup0.05/'
        
        ### Get results
        model_path = Path(args.rootdir) / folder
        _id = '_'.join(folder.split('lr')[0].split('_')[1:-1])
        if folder == 'random_baseline/':
            rankings = np.load(model_path / f'random_baseline_rankings.npy')
        else:
            rankings = np.load(model_path / f'max_new_tokens_{max_new_tokens}_rankings.npy')
        for i in range(len(test_pairs)):
            ranking = rankings[i, :TOPK]
            gt = test_pairs[i]['ground_truth_indices']
            results.append([i, ', '.join(str(x) for x in ranking), ', '.join(str(x) for x in gt)])

        # Evaluate on all GTs
        eval_metrics(rankings, test_pairs, [1, 5, 10, 20, 50, 100, 500])
        
        ### Evaluate on each GT
        _, scores = eval_on_each_gt(rankings, test_pairs, [1, 5, 10, 20, 50, 100, 500])
        
        import pandas as pd
        ### Record Results     
        score_results = pd.DataFrame(scores)
        score_results.to_csv('results/gaussian_synthetic_inf/recall_per_gt.csv', index=False)
        
        
        results = pd.DataFrame(results, columns=['query_id', 'predictions', 'ground_truth'])
        results.to_csv('results/gaussian_synthetic_inf/predictions.csv', index=False)

    elif args.command == 'get_best_model':
        # folder_list = ['sm_hn_hungarian_contrastive_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/', 
        #                'sm_hn_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_hn_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_hn_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_normalized_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_normalized_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_normalized_hungarian_contrastive_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_normalized_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/']
        # folder_list = ['sm_hn_from_stage3_hungarian_contrastive_lr5e-6_temp0.05_batch512_ep100_warmup0.05/',
        #                'sm_hn_from_stage3_contrastive_all_labels_ordered_lr5e-6_temp0.05_batch512_ep100_warmup0.05/',
        #                'sm_hn_from_stage3_contrastive_all_labels_shuffled_lr5e-6_temp0.05_batch512_ep100_warmup0.05/',
        #                'sm_hn_from_stage3_contrastive_one_label_shuffled_lr5e-6_temp0.05_batch512_ep100_warmup0.05/']
        for folder in folder_list:
            model_path = Path(args.rootdir) / folder
            all_paths = list(model_path.glob('checkpoint_*'))
            base_lms = []
            linears = []
            for path in all_paths:
                if path.is_dir():
                    base_lms.append(path.name)
                else:
                    linears.append(path.name)
            base_lms = sorted(base_lms, key=lambda x: int(x.split('_')[1]))
            linears = sorted(linears, key=lambda x: int(x.split('_')[1]))
            best_base_lm_path = model_path / base_lms[-1]
            best_linear_path = model_path / linears[-1]

            import shutil
            shutil.copytree(best_base_lm_path, model_path / 'best_model', dirs_exist_ok=True)
            shutil.copy(best_linear_path, model_path / 'best_model_linear.pt')
    elif args.command == 'delete_models':
        import shutil
        import os
        folder_list = ['sm_hn_hungarian_contrastive_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/', 
                       'sm_hn_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
                       'sm_hn_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
                       'sm_hn_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
                       'sm_normalized_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
                       'sm_normalized_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
                       'sm_normalized_hungarian_contrastive_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
                       'sm_normalized_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
                       'sm_hn_from_stage3_hungarian_contrastive_lr5e-6_temp0.05_batch512_ep100_warmup0.05/',
                       'sm_hn_from_stage3_contrastive_all_labels_ordered_lr5e-6_temp0.05_batch512_ep100_warmup0.05/',
                       'sm_hn_from_stage3_contrastive_all_labels_shuffled_lr5e-6_temp0.05_batch512_ep100_warmup0.05/',
                       'sm_hn_from_stage3_contrastive_one_label_shuffled_lr5e-6_temp0.05_batch512_ep100_warmup0.05/']
        for folder in folder_list:
            model_path = Path(args.rootdir) / folder
            all_linear_paths = list(model_path.glob('checkpoint_*.pt'))
            all_base_lm_paths = list(model_path.glob('checkpoint_*'))
            if (model_path / 'best_model').exists() and (model_path / 'best_model_linear.pt').exists():
                # print(model_path / 'best_model')
                # print(model_path / 'best_model_linear.pt')
                for path in all_linear_paths:
                    os.remove(path)
                for path in all_base_lm_paths:
                    if path.is_dir():
                        shutil.rmtree(path)

    elif args.command == 'write_test_script':
        # folder_list = ['sm_hn_hungarian_contrastive_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/', 
        #                'sm_hn_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_hn_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_hn_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_normalized_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_normalized_contrastive_one_label_shuffled_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_normalized_hungarian_contrastive_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/',
        #                'sm_normalized_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch512_ep1000_warmup0.05/']
        folder_list = ['sm_hn_from_stage3_hungarian_contrastive_lr5e-6_temp0.05_batch512_ep100_warmup0.05/',
                       'sm_hn_from_stage3_contrastive_all_labels_ordered_lr5e-6_temp0.05_batch512_ep100_warmup0.05/',
                       'sm_hn_from_stage3_contrastive_all_labels_shuffled_lr5e-6_temp0.05_batch512_ep100_warmup0.05/',
                       'sm_hn_from_stage3_contrastive_one_label_shuffled_lr5e-6_temp0.05_batch512_ep100_warmup0.05/']
        script_str = 'python test.py \\ \n --model_paths '
        for folder in folder_list:
            script_str += ' ${results_dir}/'
            script_str += folder
        script_str += ' \\ \n --data_dir data_creation/gaussian/data/opposing_pairs_data/'
        script_str += ' \\ \n --checkpoint_name best_model'
        script_str += ' \\ \n -n 1 5'
        script_str += ' \\ \n --split train'
        script_str += ' \\ \n --k_values 10 20 50 100'
        print(script_str)
        
        # python test.py
        #  ${results_dir}/sm_mse_first_label_lr5e-5_temp0.05_batch64_ep100_warmup0.05/ ${results_dir}/sm_mse_all_labels_lr5e-5_temp0.05_batch64_ep100_warmup0.05/ ${results_dir}/sm_hungarian_contrastive_lr2e-5_temp0.05_batch64_ep100_warmup0.05/ ${results_dir}/sm_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch64_ep100_warmup0.05 ${results_dir}/sm_contrastive_first_label_lr5e-5_temp0.05_batch64_ep100_warmup0.05/ ${results_dir}/sm_contrastive_one_label_shuffled_lr5e-5_temp0.05_batch64_ep100_warmup0.05/ ${results_dir}/sm_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch64_ep100_warmup0.05/ \
        # --data_dir  \
        # --checkpoint_name checkpoint_2501 \
        # -n 1 5  \
        # --split train \
        # --k_values 10 20 50 100 
    elif args.command == 'most_similar_distractors':
        max_dist_list = []
        similarities = cosine_similarity(corpus, corpus)
        for i in range(len(similarities)):
            similarities[i, i] = 0
        for pair in tqdm(test_pairs):
            gt_ids = pair['ground_truth_indices']
            for gt_id in gt_ids:
                max_dist = np.max(similarities[gt_id, :])
                max_dist_list.append(max_dist)
        max_dist_list = np.array(max_dist_list)
        print('max, min, mean, median:', np.max(max_dist_list), np.min(max_dist_list), np.mean(max_dist_list), np.median(max_dist_list))
    elif args.command == 'index_corpus':
        from test import load_model_local
        import torch
        model, _, device = load_model_local(base_model_id="meta-llama/Llama-3.2-1B-Instruct", 
                                            adapter_path='results/gaussian_synthetic_inf/sm_dual_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch128_ep400_warmup0.05/best_model/', 
                                            linear_checkpoint_path='results/gaussian_synthetic_inf/sm_dual_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch128_ep400_warmup0.05/best_model_linear.pt', 
                                            model_type="EmbeddingModelDual",
                                            embedding_model_dim=1024)
        model.to(device)
        model.eval()
        with torch.no_grad():
            batch_size = 512
            corpus_embeddings = []
            for i in range(0, len(corpus), batch_size):
                batch = corpus[i:i+batch_size]
                batch_embeddings = model.index(torch.from_numpy(batch).float().to(device))
                corpus_embeddings.append(batch_embeddings)
            corpus_embeddings = torch.cat(corpus_embeddings, dim=0)
        np.save('indexed_corpus_contrastive_all_labels_ordered.npy', corpus_embeddings.cpu().numpy())
            