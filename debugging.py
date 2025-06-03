from functools import partial
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dataset import eval_collator, load_embeddings_dataset, load_data, TrainCollator
import itertools
from tqdm import tqdm, trange

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]
    

def compute_hungarian_mse_loss(input_array, target_array):
    """
    Computes the Hungarian-matched MSE loss between two batches of sets of vectors.
    Args:
        input_array: Array of shape (batch_size, 4, 1536)
        target_array: Array of shape (batch_size, 4, 1536)
        predict_question: Whether to force matching of first embeddings
    Returns:
        Average Hungarian-matched MSE loss over the batch
    """
    from scipy.optimize import linear_sum_assignment
    
    def compute_hungarian_loss(vector_1, vector_2):
        # Compute cost matrix of MSE between all pairs
        cost_matrix = np.sum((vector_1[:, np.newaxis] - vector_2[np.newaxis, :]) ** 2, axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Gather matched pairs and compute MSE
        matched_input = vector_1[row_ind]
        matched_target = vector_2[col_ind]
        mse = np.mean((matched_input - matched_target) ** 2)
        return mse

    batch_size, k, d = input_array.shape
    losses = []
    for b in range(batch_size):
        mse = compute_hungarian_loss(input_array[b], target_array[b])
        losses.append(mse)
    return np.mean(losses)
    
def compute_max_cosine_similarity(output_doc_embeddings, label_doc_embeddings):
    # output_doc_embeddings: (batch_size, 4, 1536)
    # label_doc_embeddings: (batch_size, 4, 1536)
    max_cosine_similarities = []
    for i in range(output_doc_embeddings.shape[0]):
        max_cosine_similarity = -float('inf')
        
        # Generate all possible permutations of label indices
        label_indices = list(range(4))
        for perm in itertools.permutations(label_indices):
            # Reorder labels according to current permutation
            permuted_labels = label_doc_embeddings[i][list(perm)]
            
            # Compute cosine similarity for this permutation
            cosine = cosine_similarity(output_doc_embeddings[i], permuted_labels)
            # Get the mean of diagonal elements (matching pairs)
            mean_cosine = np.mean(np.diag(cosine))
            
            # Update maximum cosine similarity if current permutation gives better result
            if mean_cosine > max_cosine_similarity:
                max_cosine_similarity = mean_cosine
        
        max_cosine_similarities.append(max_cosine_similarity)
    return sum(max_cosine_similarities) / len(max_cosine_similarities)

def eval_top_documents(output_doc_embeddings_path, label_doc_embeddings_path):
    # def compute_cosine_similarity(output_doc_embeddings, label_doc_embeddings):
    #     cosine_loss = 0
    #     for i in range(output_doc_embeddings.shape[0]):
    #         for j in range(output_doc_embeddings.shape[1]):
    #             cosine_loss += np.mean(np.dot(output_doc_embeddings[i][j], label_doc_embeddings[i][j]) / (np.linalg.norm(output_doc_embeddings[i][j]) * np.linalg.norm(label_doc_embeddings[i][j])))
    #     cosine_loss /= (output_doc_embeddings.shape[0] * output_doc_embeddings.shape[1])
    #     return cosine_loss
    
    output_doc_embeddings = np.load(output_doc_embeddings_path)
    label_doc_embeddings = np.load(label_doc_embeddings_path)
    
    
    all_label_indices = np.arange(label_doc_embeddings.shape[0])
    # make all label indices into pairs
    shuffled_label_indices = np.random.permutation(all_label_indices)
    # compute MSE loss between two embeddings
    mse_loss = 0
    cosine_losses = []
    for i in range(0, label_doc_embeddings.shape[0], 2):
        # print(i)
        assert i % 2 == 0
        if i == label_doc_embeddings.shape[0] - 1:
            break
        # print(shuffled_label_indices[i], shuffled_label_indices[i+1], label_doc_embeddings[shuffled_label_indices[i]][0].shape, label_doc_embeddings[shuffled_label_indices[i+1]][0].shape)
        loss = np.mean((label_doc_embeddings[shuffled_label_indices[i]][0] - label_doc_embeddings[shuffled_label_indices[i+1]][0]) ** 2)
        mse_loss += loss
        for j in range(label_doc_embeddings[shuffled_label_indices[i]].shape[0]):
            cs = cosine_similarity(label_doc_embeddings[shuffled_label_indices[i]][j].reshape(1, -1), label_doc_embeddings[shuffled_label_indices[i+1]][j].reshape(1, -1))
            cosine_losses.append(cs[0][0])
        # print(label_doc_embeddings[shuffled_label_indices[i]].shape, label_doc_embeddings[shuffled_label_indices[i+1]].shape)
        # for j in range(label_doc_embeddings[shuffled_label_indices[i]].shape[0]):
            # cosine_loss += np.mean(np.dot(label_doc_embeddings[shuffled_label_indices[i]][j], label_doc_embeddings[shuffled_label_indices[i+1]][j]) / (np.linalg.norm(label_doc_embeddings[shuffled_label_indices[i]][j]) * np.linalg.norm(label_doc_embeddings[shuffled_label_indices[i+1]][j])))
    mse_loss /= i / 2
    cosine_loss = sum(cosine_losses) / len(cosine_losses)
    print(f'MSE loss: {mse_loss}')
    print(f'Cosine loss: {cosine_loss}')
    
    
    # compute the MSE loss between the output and label doc embeddings
    print('output_doc_embeddings', output_doc_embeddings.shape, 'label_doc_embeddings', label_doc_embeddings.shape)
    output_doc_embeddings = output_doc_embeddings.reshape(label_doc_embeddings.shape[0], -1, 1536)
    label_doc_embeddings = label_doc_embeddings[:,1:5]
    cosine_losses = []
    
    loss = np.mean((output_doc_embeddings - label_doc_embeddings) ** 2)
    # compute the cosine loss between the output and label doc embeddings
    print(output_doc_embeddings.shape, label_doc_embeddings.shape)
    for i in range(output_doc_embeddings.shape[0]):
        for j in range(output_doc_embeddings.shape[1]):
            cs = cosine_similarity(output_doc_embeddings[i][j].reshape(1, -1), label_doc_embeddings[i][j].reshape(1, -1))
            print(cs[0][0])
            cosine_losses.append(cs[0][0])
    cosine_loss = sum(cosine_losses) / len(cosine_losses)
    print(f'MSE loss: {loss}')
    print(f'Cosine loss: {cosine_loss}')
    
    
def compare_output_embedding_with_gt(input_data_path = 'autoregressive_wsd_train_dataset_1b',
                                     return_full_dataset = False,
                                     embeddings_path = None):
    train_collator = partial(TrainCollator(), question_only=False, shuffle=False)
    dev_collator = partial(eval_collator, question_only=False)
    full_dataset = load_embeddings_dataset(dataset_path=input_data_path)
    trainloader, _ = load_data(full_dataset, train_collator, 1, return_full_dataset = return_full_dataset, val_only = False)
    # devloader = load_data(full_dataset, dev_collator, 1, return_full_dataset = return_full_dataset, val_only = True)
    
    # embeddings = np.load(embeddings_path)
    # embeddings = embeddings.reshape(len(devloader), -1, 1536)
    # print(embeddings.shape)
    
    # Query vector & its randomly sampled target document vector 
    print('Query vector & its randomly sampled target document vector')
    hungarian_losses = []
    mcses = []
    for i, batch in enumerate(tqdm(trainloader)):
        labels = batch['labels'].cpu().numpy()
        qemb = labels[:,0]
        random_target_emb = labels[:,1]
        mse_loss = np.mean((qemb - random_target_emb) ** 2)
        cs = cosine_similarity(qemb.reshape(1, -1), random_target_emb.reshape(1, -1))
        hungarian_losses.append(mse_loss)
        mcses.append(cs[0][0])
    print(f'Average hungarian loss: {sum(hungarian_losses) / len(hungarian_losses)}')
    print(f'Average mcs: {sum(mcses) / len(mcses)}')
    
    # Query vector & target document of different query
    print('Query vector & target document of different query')
    qembs = []
    random_target_embs = []
    hungarian_losses = []
    mcses = []
    for i, batch in enumerate(tqdm(trainloader)):
        labels = batch['labels'].cpu().numpy()
        qemb = labels[:,0]
        random_target_emb = labels[:,1]
        qembs.append(qemb)
        random_target_embs.append(random_target_emb)
    
    import random
    random.shuffle(qembs)
    for i in range(len(qembs)):
        qemb = qembs[i]
        random_target_emb = random_target_embs[i]
        mse_loss = np.mean((qemb - random_target_emb) ** 2)
        cs = cosine_similarity(qemb.reshape(1, -1), random_target_emb.reshape(1, -1))
        hungarian_losses.append(mse_loss)
        mcses.append(cs[0][0])
    print(f'Average hungarian loss: {sum(hungarian_losses) / len(hungarian_losses)}')
    print(f'Average mcs: {sum(mcses) / len(mcses)}')
    
    # Two different target documents of the same query
    print('Two different target documents of the same query')
    hungarian_losses = []
    mcses = []
    for i, batch in enumerate(tqdm(trainloader)):
        labels = batch['labels'].cpu().numpy()
        random_target_emb_1 = labels[:,1]
        random_target_emb_2 = labels[:,2]
        mse_loss = np.mean((random_target_emb_1 - random_target_emb_2) ** 2)
        cs = cosine_similarity(random_target_emb_1.reshape(1, -1), random_target_emb_2.reshape(1, -1))
        hungarian_losses.append(mse_loss)
        mcses.append(cs[0][0])
    print(f'Average hungarian loss: {sum(hungarian_losses) / len(hungarian_losses)}')
    print(f'Average mcs: {sum(mcses) / len(mcses)}')
    
    
    # Two target documents belonging to two different queries
    print('Two target documents belonging to two different queries')
    
        
    target_embs = []
    random_target_embs = []
    hungarian_losses = []
    mcses = []
    for i, batch in enumerate(tqdm(trainloader)):
        labels = batch['labels'].cpu().numpy()
        target_embs.append(labels[:,1])
    
    shuffled_indices = list(range(len(target_embs)))
    import random
    random.shuffle(shuffled_indices)
    def check_shuffled_indices(shuffled_indices):
        for i in range(len(shuffled_indices)):
            if shuffled_indices[i] == i:
                return False
        return True
    while not check_shuffled_indices(shuffled_indices):
        print('not shuffled')
        random.shuffle(shuffled_indices)   
        
    for i in range(len(shuffled_indices)):
        random_target_embs.append(target_embs[shuffled_indices[i]])
    
    for i in range(len(target_embs)):
        qemb = target_embs[i]
        random_target_emb = random_target_embs[i]
        mse_loss = np.mean((qemb - random_target_emb) ** 2)
        cs = cosine_similarity(qemb.reshape(1, -1), random_target_emb.reshape(1, -1))
        hungarian_losses.append(mse_loss)
        mcses.append(cs[0][0])
    print(f'Average hungarian loss: {sum(hungarian_losses) / len(hungarian_losses)}')
    print(f'Average mcs: {sum(mcses) / len(mcses)}')



    # # indices 
    # indices = np.arange(len(dataloader))
    # shuffled_indices = np.random.permutation(indices)
    # for i, batch in enumerate(tqdm(dataloader)):
    #     labels = batch['labels'].cpu().numpy()
    #     embeddings_i = embeddings[shuffled_indices[i]].reshape(1, embeddings[shuffled_indices[i]].shape[0], embeddings[shuffled_indices[i]].shape[1])
    #     mcs = compute_max_cosine_similarity(embeddings_i, labels)
    #     hungarian_loss = compute_hungarian_mse_loss(embeddings_i, labels)
    #     hungarian_losses.append(hungarian_loss)
    #     mcses.append(mcs)
    
    # print(f'Average hungarian loss: {sum(hungarian_losses) / len(hungarian_losses)}')
    # print(f'Average mcs: {sum(mcses) / len(mcses)}')
    
    
    
    # all_labels = []
    # for batch in tqdm(dataloader):
    #     all_labels.append(batch['labels'].cpu().numpy())
    # hungarian_losses = []
    # mcses = []
    # # compute the hungarain loss and mcs between two random labels
    # for i in trange(len(all_labels)):
    #     labels = all_labels[i]
    #     # another random index
    #     j = np.random.randint(0, len(all_labels))
    #     labels_j = all_labels[j]
    #     mcs = compute_max_cosine_similarity(labels, labels_j)
    #     hungarian_loss = compute_hungarian_mse_loss(labels, labels_j)
    #     hungarian_losses.append(hungarian_loss)
    #     mcses.append(mcs)
    # print(f'Average hungarian loss: {sum(hungarian_losses) / len(hungarian_losses)}')
    # print(f'Average mcs: {sum(mcses) / len(mcses)}')
    
    
if __name__ == '__main__':
    # eval_top_documents(output_doc_embeddings_path = 'output_embeddings/qampari_org_max5_10k/out_qampari_org_max5_10k__hungarian_dev.npy',
    #                    label_doc_embeddings_path = '../Multi_Answer/mteb_retriever/doc_embeddings_qampari_org_max5_10k_retrieval_out_dev.npy')
    
    
    # data = read_jsonl('results/qampari_org_max5_10k/toy_hungarian/retrieval_out_dev.jsonl')
    # avg_score = 0
    # for inst in data:
    #     for j in range(len(inst['ctxs'])-4, len(inst['ctxs'])):
    #         avg_score += float(inst['ctxs'][j]['score'])
    # avg_score /= (len(data) * 4)
    # print(f'Average score: {avg_score}')
    
    suffix = ""
    train_name = "qampari_org_max5" # toy  toy_hungarian  toy_no_qpred  toy_no_qpred_hungarian
    dataset_name = "qampari_org_max5"
    max_new_tokens = 1
    embedding_model_dim = 1536
    
    # Create a table to store results
    
    for suffix in [""]:
        adapter_path = f"results/{train_name}/toy{suffix}/checkpoint_1000"
        # adapter_path = None
        base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
        linear_checkpoint_path = f"results/{train_name}/toy{suffix}/checkpoint_1000_linear.pt"
        dataset_path = f'training_datasets/autoregressive_{dataset_name}_train_dataset_1b'
        tf_out_embeddings_path = f'output_embeddings/{train_name}/tf_out_{train_name}_{suffix}_dev.npy'
        generated_embeddings_path = f'output_embeddings/{train_name}/out_{train_name}_{suffix}_dev.npy'
        return_full_dataset = False
    
        compare_output_embedding_with_gt(input_data_path = dataset_path,
                                         return_full_dataset = return_full_dataset,
                                         embeddings_path = generated_embeddings_path)