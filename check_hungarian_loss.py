from model import HungarianContrastiveLoss, ContrastiveLoss
from dataset import load_embeddings_dataset, DataHandler, MSETrainCollator, ContrastiveTrainCollator
import torch
from tqdm import trange
# input_ids torch.Size([5, 257])
# attention_mask torch.Size([5, 257])
# positive_embeddings torch.Size([5, 3, 1536])
# negative_embeddings torch.Size([5, 3, 1536])

# fix random seed
torch.manual_seed(0)


def permute_batch(batch):
    # permute the batch
    # only permute the positive embeddings
    # permute the order of each item in the batch
    for i in range(batch['positive_embeddings'].size(0)):
        batch['positive_embeddings'][i] = batch['positive_embeddings'][i][torch.randperm(batch['positive_embeddings'][i].size(0))]
    return batch


def check_hungarian_loss():
    # positive_embeddings: Tensor of shape (batch_size, k, d)
    # negative_embeddings: Tensor of shape (batch_size, k, d)
    
    # load data
    full_dataset = load_embeddings_dataset(dataset_path="training_datasets/autoregressive_ambiguous_inf_train_dataset_1b_contrastive_5_ctxs/")
    collator = ContrastiveTrainCollator()
    data_handler = DataHandler(full_dataset, collator, 4, 'train')
    train_dataloader, valid_loss_dataloader = data_handler.get_train_dev_dataloader(random_train_loader=False)
    
    for batch in valid_loss_dataloader:
        # get a random batch outputs that is of the same shape as the positive_embeddings and negative_embeddings
        output = torch.randn(batch['positive_embeddings'].size())
        batch['outputs'] = output
        
        # compute loss
        loss_fn_h = HungarianContrastiveLoss()
        loss_fn = ContrastiveLoss()
        eps = 1e-6
        for _ in range(10000):
            perm_batch = permute_batch(batch)
            loss_h = loss_fn_h(perm_batch['outputs'], perm_batch['positive_embeddings'], perm_batch['negative_embeddings'])
            loss = loss_fn(perm_batch['outputs'], perm_batch['positive_embeddings'], perm_batch['negative_embeddings'])
            print(loss_h, loss, loss_h <= loss + eps)
            if loss_h > loss + eps:
                print('-'*100)
        break

    
    
    
    
    
    
if __name__ == "__main__":
    check_hungarian_loss()