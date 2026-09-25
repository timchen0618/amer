import torch
from datasets import load_dataset
import numpy as np

import torch.nn as nn
import torch.optim as optim

# Load dataset
# Load custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embeddings = torch.tensor(self.embeddings[idx], dtype=torch.float)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"embeddings":embeddings, "labels":labels}
    
    
def load_data(regression=False):
    # Example embeddings and labels
    data = {}
    for split in ['train', 'dev', 'test']:
        data[split] = {}

    for split in ['train', 'test']:
        data[split]['embeddings'] = np.load(f'qembs_{split}.npy')
        if regression:    
            data[split]['labels'] = np.load(f'{split}_lengths_actual.npy')
        else:
            data[split]['labels'] = np.load(f'{split}_lengths.npy')
        
        full_dataset = CustomDataset(data[split]['embeddings'], data[split]['labels'])
        if split == 'train':
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            # Split the dataset
            data['train']['dataset'], data['dev']['dataset'] = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        else:
            data[split]['dataset'] = full_dataset
    return data