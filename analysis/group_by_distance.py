import torch
from torch.utils.data import DataLoader
from dataset import (
    load_embeddings_dataset,
    contrastive_eval_collator,
    mse_eval_collator,
    DataHandler
)
from functools import partial
import structlog
logger = structlog.get_logger()

def load_input_data(loss_function, question_only, batch_size_training, get_split, input_data_path):
    use_contrastive_data = 'Contrastive' in loss_function
    logger.info('use_contrastive_data', more_than_strings=use_contrastive_data)
    # Load dataset
    if use_contrastive_data:
        collator = contrastive_eval_collator
    else:
        collator = partial(mse_eval_collator, question_only=question_only)
    full_dataset = load_embeddings_dataset(dataset_path=input_data_path)
    data_handler = DataHandler(full_dataset, collator, batch_size_training, 'train' if 'train' in get_split else 'dev')
    
    # load the corresponding split
    if get_split == 'train-held-out':
        dataloader = data_handler.get_dev_dataloader()
    elif get_split == 'train':
        dataloader = data_handler.get_sequential_train_dataloader()
    elif get_split == 'dev':
        dataloader = data_handler.get_full_dataloader()
    else:
        raise ValueError(f"Invalid split: {get_split}")
    return dataloader


def write_list_to_file(file_path, list_of_indices):
    with open(file_path, 'w') as f:
        for index in list_of_indices:
            f.write(f"{index}\n")

# load_input_data(loss_function, question_only, batch_size_training, get_split, input_data_path):
data_name = 'ambiguous_qe'
retriever = 'inf'
dataset_name = f"{data_name}_{retriever}"

dataset_path = f'training_datasets/{data_name}/{retriever}/autoregressive_{dataset_name}_dev_dataset_1b_contrastive_2_to_5_ctxs'
dataloader = load_input_data('Hungarian_Contrastive', False, 1, 'dev', dataset_path)


def average_pairwise_distances(embeddings):
    distances = torch.cdist(embeddings, embeddings)
    distances = torch.triu(distances, diagonal=1)
    if distances[distances.nonzero(as_tuple=True)].numel() == 0:
        return 0
    return torch.mean(distances[distances.nonzero(as_tuple=True)]).item()

avg_distances = []
large_distance_indices = []
small_distance_indices = []
for i, batch in enumerate(dataloader):
    avg_distances.append(average_pairwise_distances(batch['positive_embeddings']))

avg_distances = torch.tensor(avg_distances)
quantile_1 = torch.quantile(avg_distances, 1/3, interpolation='nearest')
quantile_2 = torch.quantile(avg_distances, 2/3, interpolation='nearest')
logger.info('avg_distances', quantile_1=quantile_1, quantile_2=quantile_2)

for i, batch in enumerate(dataloader):
    if average_pairwise_distances(batch['positive_embeddings']) > quantile_2:
        large_distance_indices.append(i)
    if average_pairwise_distances(batch['positive_embeddings']) < quantile_1:
        small_distance_indices.append(i)

print(len(large_distance_indices))
print(len(small_distance_indices))
# print(sorted(avg_distances))



write_list_to_file(f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/qampari_embeddings_data/large_distance_indices_{retriever}.txt', large_distance_indices)
write_list_to_file(f'/scratch/cluster/hungting/projects/autoregressive/data/ambiguous/qampari_embeddings_data/small_distance_indices_{retriever}.txt', small_distance_indices)

# write_list_to_file(f'/scratch/cluster/hungting/projects/autoregressive/data/qampari/large_distance_indices_{retriever}.txt', large_distance_indices)
# write_list_to_file(f'/scratch/cluster/hungting/projects/autoregressive/data/qampari/small_distance_indices_{retriever}.txt', small_distance_indices)
