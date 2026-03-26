from pathlib import Path
import json
import numpy as np
    
"""
    The script is used to evaluate the accuracy of the predicted length labels.
    The training is to predict "n <embed>", and then n should match the number of ground truth docs.
"""

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def write_jsonl(file_path, data):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def get_data_mapping(project_root):
    """Returns the data mapping configuration."""
    return {
        "qampari": {
            'dev': f'{project_root}/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl',
            'train': f'{project_root}/diverse_response/data/qampari_data/train_data_gt_qampari_corpus.jsonl',
            'second_stage': f'{project_root}/diverse_response/data/qampari_data/2nd_stage_test_data/dev_data_qampari_corpus_inp{{num_input}}.jsonl'
        },
        "qampari_5_to_8": {
            'dev_5_to_8': f'{project_root}/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus_5_to_8_ctxs.jsonl',
        },
        "ambiguous": {
            'dev': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_no_empty_clusters.jsonl',
            'dev_2_to_5': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.jsonl',
            'train': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data',
        },
        "ambiguous_qe": {
            'dev': f'{project_root}/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev.jsonl',
            'dev_2_to_5': f'{project_root}/autoregressive/data/ambiguous/qampari_embeddings_data/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.jsonl',
            'train': f'{project_root}/autoregressive/data/ambiguous/nq_embeddings_data',
        },
        "wsd_distinct": {
            'dev': f'{project_root}/autoregressive/data/wsd/distinct/dev.jsonl',
        },
    }

rootdir = Path('/scratch/hc3337/projects/autoregressive/results/llama-1b/qampari+ambiguous_qe_inf/fixed_model/')
data_mapping = get_data_mapping('/scratch/hc3337/projects')
data_path = data_mapping['qampari_5_to_8']['dev_5_to_8']
suffix_list = [
    "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1", 
    "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1",
    "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_hungarian_contrastive_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1"
]

retrieved_docs = read_jsonl(data_path)
length_label = np.array([len(doc['ground_truths']) for doc in retrieved_docs])
# compute majority label occurrence
majority_label = np.array([np.bincount(length_label).argmax()])
majority_label_occurrence = 100*np.bincount(length_label).max() / len(length_label)
print(f'majority label: {majority_label}, majority label occurrence: {majority_label_occurrence}')

for suffix in suffix_list:
    predicted_length = np.load(rootdir / suffix / "retrieval_out_dev_qampari_5_to_8_lengths.npy")
    assert len(retrieved_docs) == len(predicted_length)
    acc = 0
    for i in range(len(retrieved_docs)):
        if predicted_length[i] == length_label[i]:
            acc += 1
    print(f'{suffix} accuracy: {100*acc / len(retrieved_docs)}')
    


data_mapping = get_data_mapping('/scratch/hc3337/projects')
data_path = data_mapping['ambiguous_qe']['dev_2_to_5']
suffix_list = [
    "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1", 
    "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1",
    "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_hungarian_contrastive_lr2e-5_temp0.05_batch16_ep120_warmup0.05_srm1"
]

retrieved_docs = read_jsonl(data_path)
if 'ground_truths' in retrieved_docs[0]:
    length_label = np.array([len(doc['ground_truths']) for doc in retrieved_docs])
else:
    length_label = np.array([len(doc['positive_ctxs']) for doc in retrieved_docs])
# compute majority label occurrence
majority_label = np.array([np.bincount(length_label).argmax()])
majority_label_occurrence = 100*np.bincount(length_label).max() / len(length_label)
print(f'majority label: {majority_label}, majority label occurrence: {majority_label_occurrence}')

for suffix in suffix_list:
    predicted_length = np.load(rootdir / suffix / "retrieval_out_dev_ambiguous_qe_lengths.npy")
    assert len(retrieved_docs) == len(predicted_length)
    acc = 0
    for i in range(len(retrieved_docs)):
        if predicted_length[i] == length_label[i]:
            acc += 1
    print(f'{suffix} accuracy: {100*acc / len(retrieved_docs)}')
    
    
    
rootdir = Path('/scratch/hc3337/projects/autoregressive/results/llama-1b/qampari_inf/fixed_model/')
data_mapping = get_data_mapping('/scratch/hc3337/projects')
data_path = data_mapping['qampari_5_to_8']['dev_5_to_8']
suffix_list = [
    "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_ordered_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10",
    "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_shuffled_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10",
    "normalized_qampari_4gpu_full_finetuning_SSVariableLeftPadPredLength_hungarian_contrastive_lr2e-5_temp0.05_batch32_ep120_warmup0.05_srm10"
]

retrieved_docs = read_jsonl(data_path)
length_label = np.array([len(doc['ground_truths']) for doc in retrieved_docs])
# compute majority label occurrence
majority_label = np.array([np.bincount(length_label).argmax()])
majority_label_occurrence = 100*np.bincount(length_label).max() / len(length_label)
print(f'majority label: {majority_label}, majority label occurrence: {majority_label_occurrence}')

for suffix in suffix_list:
    predicted_length = np.load(rootdir / suffix / "retrieval_out_dev_qampari_5_to_8_lengths.npy")
    assert len(retrieved_docs) == len(predicted_length)
    acc = 0
    for i in range(len(retrieved_docs)):
        if predicted_length[i] == length_label[i]:
            acc += 1
    print(f'{suffix} accuracy: {100*acc / len(retrieved_docs)}')
    

rootdir = Path('/scratch/hc3337/projects/autoregressive/results/llama-1b/ambiguous_qe_inf/fixed_model/')    
data_mapping = get_data_mapping('/scratch/hc3337/projects')
data_path = data_mapping['ambiguous_qe']['dev_2_to_5']
suffix_list = [
    "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_ordered_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1",
    "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPadPredLength_contrastive_all_labels_shuffled_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1",
    "normalized_ambiguous_qe_4gpu_full_finetuning_SSVariableLeftPadPredLength_hungarian_contrastive_lr5e-5_temp0.05_batch32_ep120_warmup0.05_srm1"
]

retrieved_docs = read_jsonl(data_path)
if 'ground_truths' in retrieved_docs[0]:
    length_label = np.array([len(doc['ground_truths']) for doc in retrieved_docs])
else:
    length_label = np.array([len(doc['positive_ctxs']) for doc in retrieved_docs])
# compute majority label occurrence
majority_label = np.array([np.bincount(length_label).argmax()])
majority_label_occurrence = 100*np.bincount(length_label).max() / len(length_label)
print(f'majority label: {majority_label}, majority label occurrence: {majority_label_occurrence}')

for suffix in suffix_list:
    predicted_length = np.load(rootdir / suffix / "retrieval_out_dev_ambiguous_qe_lengths.npy")
    assert len(retrieved_docs) == len(predicted_length)
    acc = 0
    for i in range(len(retrieved_docs)):
        if predicted_length[i] == length_label[i]:
            acc += 1
    print(f'{suffix} accuracy: {100*acc / len(retrieved_docs)}')