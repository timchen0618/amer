import json
import numpy as np

def read_jsonl(file_path):
    """
    Read a JSONL file and return a list of dictionaries.

    :param file_path: Path to the JSONL file
    :return: List of dictionaries
    """
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    
train_data = read_jsonl('/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/train_data_gt_qampari_corpus.jsonl')
dev_data = read_jsonl('/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/dev_data_gt_qampari_corpus.jsonl')

def check_duplicates(data):
    deduped_lengths = []
    deduped_actual_lengths = []
    for inst in data:
        dedup_gts = []
        dedup_ids = set()

        gts = inst['ground_truths']
        for gt in gts:
            if gt['id'] not in dedup_ids:
                dedup_gts.append(gt)
                dedup_ids.add(gt['id'])
        assert len(dedup_gts) > 0
        if len(dedup_gts) > 10:
            deduped_lengths.append(10)
        else:
            deduped_lengths.append(len(dedup_gts)-1)
        deduped_actual_lengths.append(len(dedup_gts))
    return np.array(deduped_lengths), np.array(deduped_actual_lengths)
    
def check_duplicates_clusters(data):
    duplicates = 0
    deduped_lengths = []
    deduped_actual_lengths = []
    for inst in data:
        dedup_gts = []
        dedup_ids = set()

        gts = inst['ground_truths']
        for cluster in gts:
            cluster_ids = ''.join([doc['id'] for doc in cluster])
            if cluster_ids not in dedup_ids:
                dedup_gts.append(cluster)
                dedup_ids.add(cluster_ids)
        if len(gts) != len(dedup_gts):
            duplicates += 1
        assert len(dedup_gts) > 0
        if len(dedup_gts) > 10:
            deduped_lengths.append(10)
        else:
            deduped_lengths.append(len(dedup_gts)-1)
        deduped_actual_lengths.append(len(dedup_gts))
    return np.array(deduped_lengths), np.array(deduped_actual_lengths)
    
    
train_lengths, train_actual_lengths = check_duplicates(train_data)
dev_lengths, dev_actual_lengths = check_duplicates_clusters(dev_data)

# import matplotlib.pyplot as plt

# # Plot histogram for train_lengths
# plt.figure(figsize=(10, 5))
# plt.hist(train_lengths, bins=range(0, 12), edgecolor='black', alpha=0.7)
# plt.title('Distribution of Train Lengths')
# plt.xlabel('Length')
# plt.ylabel('Frequency')
# plt.xticks(range(1, 12))
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig('train_lengths_distribution.png')
# plt.close()

# # Plot histogram for dev_lengths
# plt.figure(figsize=(10, 5))
# plt.hist(dev_lengths, bins=range(0, 12), edgecolor='black', alpha=0.7)
# plt.title('Distribution of Dev Lengths')
# plt.xlabel('Length')
# plt.ylabel('Frequency')
# plt.xticks(range(1, 12))
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig('dev_lengths_distribution.png')
# plt.close()

print(train_actual_lengths.mean())
print(dev_actual_lengths.mean())

# np.save('train_lengths', train_lengths)
# np.save('test_lengths', dev_lengths)

# print(train_actual_lengths)
# print(dev_actual_lengths)
# np.save('train_lengths_actual', train_actual_lengths)
# np.save('test_lengths_actual', dev_actual_lengths)