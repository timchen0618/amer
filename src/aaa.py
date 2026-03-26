from src.model import EmbeddingModel, load_model
from src.dataset import (
    load_embeddings_dataset,
    ContrastiveTrainCollator,
    DataHandler,
    contrastive_eval_collator
)


import structlog
logger = structlog.get_logger()
import numpy as np



def load_input_data(input_data_path, use_ground_truth_for_eval=False):
    # Load dataset
    if use_ground_truth_for_eval:
        collator = ContrastiveTrainCollator()
    else:
        collator = contrastive_eval_collator
    full_dataset = load_embeddings_dataset(dataset_path=input_data_path)
    
    if use_ground_truth_for_eval:
        data_handler = DataHandler(full_dataset, collator, 128, 'dev', 4)
    else:
        data_handler = DataHandler(full_dataset, collator, 1, 'dev', 4)
    
    dataloader = data_handler.get_full_dataloader()
    return dataloader


dataloader = load_input_data(f'training_datasets/llama-1b/gaussian_new_mlps_rotation/inf/gaussian_new_mlps_rotation_dev_dataset_1b_contrastive_pred_length/', use_ground_truth_for_eval=False)

for batch in dataloader:
    print(batch['length_labels_input_ids'])