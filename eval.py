import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from model import load_model
from prettytable import PrettyTable

from dataset import (
    load_embeddings_dataset,
    contrastive_eval_collator,
    mse_eval_collator,
    DataHandler
)
from utils import Config, set_seed, set_optim
import yaml
from pathlib import Path

# Tokenize dataset
def tokenize_function(examples, instruction, tokenizer):
    if 'question' in examples:
        question = examples['question']
    else:
        question = examples['question_text']
    examples['text'] = formulate_text(instruction, question)
    print(examples['text'][0])
    return tokenizer(examples['text'], padding='longest', truncation=True, max_length=128, return_tensors='pt')

def formulate_text(instruction, queries):
    return [instruction.replace('[QUERY]', query) for query in queries]


def evaluate_loop(dataloader, model, device, max_new_tokens, use_gt_q_embed, compute_loss = True):

    all_outputs = []
    all_losses = []
    all_labels = []
    all_lengths = []
    adaptive_max_new_tokens = (max_new_tokens is None)
        
    for batch in tqdm(dataloader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        if 'positive_embeddings' in batch and adaptive_max_new_tokens:
            max_new_tokens = batch['positive_embeddings'].size(1)
        # print('max_new_tokens', max_new_tokens, 'positive_embeddings', batch['positive_embeddings'].size(), 'negative_embeddings', batch['negative_embeddings'].size())
        output = model.generate(
            max_new_tokens=max_new_tokens,
            use_gt_q_embed=use_gt_q_embed,
            **batch
        )
        all_outputs.append(output.view(-1, output.size(-1)))
        # print('output', output.size(), 'positive_embeddings', batch['positive_embeddings'].size(), 'negative_embeddings', batch['negative_embeddings'].size())
        if compute_loss:
            if 'labels' in batch:
                # compute the loss
                # print(output.size(), batch['labels'].size())
                assert output.size() == batch['labels'].size(), (output.size(), batch['labels'].size())
                loss = model.loss_fct(output.float(), batch['labels'].float())
                # print('loss', loss.item())
                all_lengths.append(batch['labels'].size(1))
                all_losses.append(loss.item())
                all_labels.append(batch['labels'].view(-1, batch['labels'].size(-1)))
            elif 'positive_embeddings' in batch:
                # print(output.size(), batch['positive_embeddings'].size(), batch['negative_embeddings'].size())
                loss = model.loss_fct(output.float(), batch['positive_embeddings'].float(), batch['negative_embeddings'].float())
                all_lengths.append(batch['positive_embeddings'].size(1))
                all_labels.append(batch['positive_embeddings'].view(-1, batch['positive_embeddings'].size(-1)))
                all_losses.append(loss.item())
    
    if compute_loss:
        return torch.cat(all_outputs, dim=0).cpu().numpy(), sum(all_losses) / len(all_losses), torch.cat(all_labels, dim=0).cpu().numpy(), all_lengths
    else:
        return torch.cat(all_outputs, dim=0).cpu().numpy(), None, None, all_lengths



def eval_with_generation(input_data_path = 'autoregressive_wsd_train_dataset_1b',
         get_split = 'train-held-out',
         adapter_path = "results/test/toy/checkpoint_4", 
         base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct",
         linear_checkpoint_path = None,
         output_path = 'out_train_large.npy',
         output_lengths_path = 'out_train_large_lengths.npy',
         max_new_tokens = 2,
         embedding_model_dim = 1536,
         compute_loss = True):
    
    print('loading data from ', input_data_path)
    instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    instruction = f'{instruction_template}Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: [QUERY]\n{response_template}'
    
    with open('configs/eval.yaml') as f:
        config_dict = yaml.safe_load(f)

    print("Config:", config_dict)
    configs = Config(config_dict)
    
    use_contrastive_data = 'Contrastive' in configs.loss_function
    # Load dataset
    if use_contrastive_data:
        collator = contrastive_eval_collator
    else:
        collator = partial(mse_eval_collator, question_only=configs.question_only)
    full_dataset = load_embeddings_dataset(dataset_path=input_data_path)
    data_handler = DataHandler(full_dataset, collator, configs.batch_size_training, 'train' if 'train' in get_split else 'dev')
    
    # load the corresponding split
    if get_split == 'train-held-out':
        dataloader = data_handler.get_dev_dataloader()
    elif get_split == 'train':
        dataloader = data_handler.get_sequential_train_dataloader()
    elif get_split == 'dev':
        dataloader = data_handler.get_full_dataloader()
    else:
        raise ValueError(f"Invalid split: {get_split}")

    # # Define model and tokenizer
    base_model_id = base_model_id  # Replace with your base model ID
    adapter_path = adapter_path  # Path to your saved adapter weights
    model, tokenizer = load_model(train_lora=False,
                                  base_model_id=base_model_id, 
                                  adapter_path=adapter_path, 
                                  linear_checkpoint_path=linear_checkpoint_path,
                                  embedding_model_dim=embedding_model_dim, 
                                  loss_function=configs.loss_function)
    # always use hungarian loss for evaluation
    # always not predict question for evaluation
    tokenizer.pad_token = tokenizer.eos_token

    # # tokenize dataset => get the question embedding
    # tokenized_datasets = dataset.map(partial(tokenize_function, instruction=instruction, tokenizer=tokenizer), batched=True)

    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        outputs, loss, all_labels, all_lengths = evaluate_loop(dataloader, model, device, max_new_tokens=max_new_tokens, use_gt_q_embed=configs.use_gt_q_embed, compute_loss=compute_loss)
        # outputs = outputs.reshape(-1, embedding_model_dim)
        np.save(output_path, outputs)
        if len(all_lengths) > 0:
            np.save(output_lengths_path, np.array(all_lengths))
        else:
            print('no lengths')
        return outputs, loss, all_labels
    
    

def eval_with_teacher_forcing(adapter_path = "results/test/toy-autoreg/checkpoint_4", 
              base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct", 
              dataset_path = 'autoregressive_wsd_train_dataset_1b',
              linear_checkpoint_path = None,
              output_path = 'out_train_large.npy',
              return_full_dataset = False,
              embedding_model_dim = 1536):
    # dataset_train = load_embeddings_dataset(dataset_path='autoregressive_wsd_dev_dataset')
    # load the configuration file
    with open('configs/eval.yaml') as f:
        config_dict = yaml.safe_load(f)

    print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ########
    ##### Model Loading 
    ########
    instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    instruction = f'{instruction_template}Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: [QUERY]\n{response_template}'

    # # Define model and tokenizer
    model, tokenizer = load_model(train_lora=False,
                                  base_model_id=base_model_id, 
                                  adapter_path=adapter_path, 
                                  linear_checkpoint_path=linear_checkpoint_path,
                                  embedding_model_dim=embedding_model_dim, 
                                  weight_tying=configs.weight_tying, 
                                  loss_function=configs.loss_function,
                                  predict_question=configs.predict_question)
    tokenizer.pad_token = tokenizer.eos_token


    # data loading
    print('question only', configs.question_only)
    collator = partial(TrainCollator(), shuffle=False, question_only=configs.question_only)
    full_dataset = load_embeddings_dataset(dataset_path=dataset_path)
    valid_loss_dataloader = load_data(full_dataset, collator, configs.batch_size_training, return_full_dataset = return_full_dataset, val_only = True)
    
    model = model.to(device)
    total_loss = 0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        model.eval()
        for step, batch in enumerate(tqdm(valid_loss_dataloader)):
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            print('loss', loss.item())
            all_outputs.append(outputs.last_hidden_states)
            all_labels.append(batch['labels'])
            # print('soize', outputs.last_hidden_states.size())
        outputs = torch.cat(all_outputs, dim=0).cpu().numpy()
        outputs = outputs.reshape(-1, embedding_model_dim)
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        np.save(output_path, outputs)
        return outputs, total_loss / len(valid_loss_dataloader), all_labels
    
    


def check_hungarian_contrastive_loss(adapter_path = "results/test/toy-autoreg/checkpoint_4", 
              base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct", 
              dataset_path = 'autoregressive_wsd_train_dataset_1b',
              linear_checkpoint_path = None,
              embedding_model_dim = 1536):
    # load the configuration file
    with open('configs/eval.yaml') as f:
        config_dict = yaml.safe_load(f)

    print("Config:", config_dict)

    configs = Config(config_dict)
    set_seed(configs.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ########
    ##### Model Loading 
    ########
    instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    instruction = f'{instruction_template}Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: [QUERY]\n{response_template}'

    # # Define model and tokenizer
    model_c, tokenizer = load_model(train_lora=False,
                                  base_model_id=base_model_id, 
                                  adapter_path=adapter_path, 
                                  linear_checkpoint_path=linear_checkpoint_path,
                                  embedding_model_dim=embedding_model_dim, 
                                  weight_tying=configs.weight_tying, 
                                  loss_function='Hungarian_Contrastive',
                                  predict_question=configs.predict_question)
    tokenizer.pad_token = tokenizer.eos_token


    # data loading
    print('question only', configs.question_only)
    collator = partial(TrainCollator(), shuffle=False, question_only=configs.question_only)
    full_dataset = load_embeddings_dataset(dataset_path=dataset_path)
    valid_loss_dataloader = load_data(full_dataset, collator, configs.batch_size_training, return_full_dataset = False, val_only = True)
    
    model_c = model_c.to(device)
    losses_c = []
    with torch.no_grad():
        model_c.eval()
        for step, batch in enumerate(tqdm(valid_loss_dataloader)):
            for k, v in batch.items():
                batch[k] = v.to(device)
            outputs_c = model_c(**batch)
            loss_c = outputs_c.loss
            losses_c.append(loss_c.item())
            if step > 100:
                break
    print('losses_c', losses_c)
            
    #  # # Define model and tokenizer
    # model_h, tokenizer = load_model(train_lora=False,
    #                               base_model_id=base_model_id, 
    #                               adapter_path=adapter_path, 
    #                               linear_checkpoint_path=linear_checkpoint_path,
    #                               embedding_model_dim=embedding_model_dim, 
    #                               weight_tying=configs.weight_tying, 
    #                               loss_function='Hungarian_Contrastive',
    #                               predict_question=configs.predict_question)
    # tokenizer.pad_token = tokenizer.eos_token


    # # data loading
    # print('question only', configs.question_only)
    # collator = partial(TrainCollator(), shuffle=False, question_only=configs.question_only)
    # full_dataset = load_embeddings_dataset(dataset_path=dataset_path)
    # valid_loss_dataloader = load_data(full_dataset, collator, configs.batch_size_training, return_full_dataset = False, val_only = True)
    
    # model_h = model_h.to(device)
    # losses_h = []
    # with torch.no_grad():
    #     model_h.eval()
    #     for step, batch in enumerate(tqdm(valid_loss_dataloader)):
    #         for k, v in batch.items():
    #             batch[k] = v.to(device)
    #         outputs_h = model_h(**batch)
    #         loss_h = outputs_h.loss
    #         losses_h.append(loss_h.item())
    #         if step > 100:
    #             break
            
    # for i in range(len(losses_c)):
    #     if losses_c[i] != losses_h[i]:
    #         print(f"Loss mismatch at index {i}")
    #         print(f"Loss 1: {losses_c[i]}")
    #         print(f"Loss 2: {losses_h[i]}")
    #         break
    
    
    
 
def check_hungarian_loss(input_data_path = 'autoregressive_wsd_train_dataset_1b',
         return_full_dataset = False,
         adapter_path = "results/test/toy/checkpoint_4", 
         base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct",
         linear_checkpoint_path = None,
         output_path = 'out_train_large.npy',
         max_new_tokens = 2,
         embedding_model_dim = 1536):
    
    print('loading data from ', input_data_path)
    instruction_template = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    instruction = f'{instruction_template}Retrieve a diverse set of documents that covers multiple aspect of the query.\nQuery: [QUERY]\n{response_template}'

    # Load dataset
    collator = eval_collator
    full_dataset = load_embeddings_dataset(dataset_path=input_data_path)
    dataloader = load_data(full_dataset, collator, 1, return_full_dataset = return_full_dataset, val_only = True)

    # # Define model and tokenizer
    base_model_id = base_model_id  # Replace with your base model ID
    adapter_path = adapter_path  # Path to your saved adapter weights
    model, tokenizer = load_model(train_lora=False,
                                  base_model_id=base_model_id, 
                                  adapter_path=adapter_path, 
                                  linear_checkpoint_path=linear_checkpoint_path,
                                  embedding_model_dim=embedding_model_dim, 
                                  use_hungarian_loss=True, 
                                  predict_question=False)
    # always use hungarian loss for evaluation
    # always not predict question for evaluation
    tokenizer.pad_token = tokenizer.eos_token

    # # tokenize dataset => get the question embedding
    # tokenized_datasets = dataset.map(partial(tokenize_function, instruction=instruction, tokenizer=tokenizer), batched=True)
    def compute_min_pairwise_mse(outputs, labels):
        import itertools
        """
        Compute minimum pairwise MSE loss between outputs and labels by enumerating all possible pairs.
        
        Args:
            outputs: Tensor of shape (1, 4, 1536)
            labels: Tensor of shape (1, 4, 1536)
            
        Returns:
            Scalar tensor: minimum pairwise MSE loss
        """
        # Remove batch dimension
        outputs = outputs.squeeze(0)  # (4, 1536)
        labels = labels.squeeze(0)    # (4, 1536)
        
        # Generate all possible permutations of label indices
        label_indices = list(range(4))
        min_mse = float('inf')
        
        # Try all possible permutations of label indices
        for perm in itertools.permutations(label_indices):
            # Reorder labels according to current permutation
            permuted_labels = labels[list(perm)]
            
            # Compute MSE for this permutation
            mse = torch.mean((outputs - permuted_labels) ** 2)
            
            # Update minimum MSE if current permutation gives better result
            if mse < min_mse:
                min_mse = mse
        
        return min_mse

    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        all_outputs = []
        all_losses = []
        for batch in tqdm(dataloader):
            for k, v in batch.items():
                batch[k] = v.to(device)

            output = model.generate(
                max_new_tokens=max_new_tokens,
                **batch
            )
            all_outputs.append(output)
            if 'labels' in batch:
                # compute the loss
                # print(output.size(), batch['labels'].size())
                assert output.size() == batch['labels'].size(), (output.size(), batch['labels'].size())
                loss = model.loss_fct(output.float(), batch['labels'].float())
                min_mse = compute_min_pairwise_mse(output.float(), batch['labels'].float())
                if loss.half().item() != min_mse.half().item():
                    print(f"Loss mismatch: {loss.item()} != {min_mse.item()}", loss.item(), min_mse.item())
                
                all_losses.append(loss.item())
    


if __name__ == "__main__":
    
    # suffix = ""
    # train_name = "qampari_org_max5" # toy  toy_hungarian  toy_no_qpred  toy_no_qpred_hungarian
    # dataset_name = "qampari_inf"
    # max_new_tokens = 1
    # embedding_model_dim = 1536
    
    # # Create a table to store results
    # results_table = PrettyTable()
    # results_table.field_names = ["Model", "Loss w/ TF", "Loss w/ Gen"]
    
    # for suffix in ["", "_hungarian"]:
    # # for suffix in ["_qemb"]:
    #     adapter_path = f"results/{train_name}/toy{suffix}/checkpoint_1000"
    #     # adapter_path = None
    #     base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    #     linear_checkpoint_path = f"results/{train_name}/toy{suffix}/checkpoint_1000_linear.pt"
    #     dataset_path = f'training_datasets/autoregressive_{dataset_name}_train_dataset_1b'
    #     tf_out_embeddings_path = f'output_embeddings/{train_name}/tf_out_{train_name}_{suffix}_train.npy'
    #     generated_embeddings_path = f'output_embeddings/{train_name}/out_{train_name}_{suffix}_train.npy'
    #     return_full_dataset = False
        
    #     print('-'*50)
    #     print(f'evaluating {suffix} on {train_name} dev set')
    #     outputs, loss_with_teacher_forcing = eval_with_teacher_forcing(adapter_path = adapter_path, 
    #             base_model_id = base_model_id, 
    #             dataset_path = dataset_path,
    #             linear_checkpoint_path = linear_checkpoint_path,
    #             output_path = tf_out_embeddings_path,
    #             return_full_dataset = return_full_dataset,
    #             embedding_model_dim = embedding_model_dim)
    #     print('writing to ', tf_out_embeddings_path)
    #     outputs = np.load(tf_out_embeddings_path)
    #     print('tf out embeddings shape', outputs.shape)
    
    #     outputs, loss_with_generation = eval_with_generation(adapter_path = adapter_path,
    #         base_model_id = base_model_id,
    #         linear_checkpoint_path = linear_checkpoint_path,
    #         output_path = generated_embeddings_path,
    #         input_data_path = dataset_path, 
    #         return_full_dataset = return_full_dataset,
    #         max_new_tokens = max_new_tokens,
    #         embedding_model_dim = embedding_model_dim)
    #     print('writing to ', generated_embeddings_path)
    #     outputs = np.load(generated_embeddings_path)
    #     print('generated embeddings shape', outputs.shape)
        
    #     # Add results to the table
    #     model_name = f"{train_name}{suffix}"
    #     results_table.add_row([model_name, f"{loss_with_teacher_forcing:.4f}", f"{loss_with_generation:.4f}"])
    
    # # Print the final results table
    # print("\nEvaluation Results:")
    # print(results_table)
    
    
    # suffix = ""
    # train_name = "msmarco_stella" # toy  toy_hungarian  toy_no_qpred  toy_no_qpred_hungarian
    # dataset_name = "msmarco_stella"
    # max_new_tokens = 2
    # embedding_model_dim = 1024
    
    num_map = {
        "msmarco_stella": 10000,
        "msmarco_cont": 10000,
        "msmarco_inf": 10000,
        "nq_cont": 70000,
        "nq_inf": 70000,
        "nq_stella": 70000,
        "ambiguous_cont": 1000,
        "ambiguous_inf": 3500,
        "ambiguous_stella": 1000,
        "qampari_cont": 30000,
        "qampari_inf": 300,
        "qampari_stella": 30000,
        "qampari+ambiguous_inf": 10000
    }
    # for train_name in ["msmarco_stella", "msmarco_cont", "msmarco_inf"]: 
    # for train_name in ["nq_cont", "nq_inf", "nq_stella"]:
    for train_name in ["nq_inf"]:
    # for train_name in ["ambiguous_cont", "ambiguous_inf", "ambiguous_stella"]:
    # for train_name in ["qampari_cont", "qampari_inf", "qampari_stella"]:
        dataset_name = "ambiguous_" + train_name.split("_")[1]
        # dataset_name = "qampari_" + train_name.split("_")[1]
        max_new_tokens = None
        if "stella" in train_name:
            embedding_model_dim = 1024
        elif "cont" in train_name:
            embedding_model_dim = 768
        elif "inf" in train_name:
            embedding_model_dim = 1536
        else:
            embedding_model_dim = 0
    
        # Create a table to store results
        results_table = PrettyTable()
        # results_table.field_names = ["Model", "Loss w/ TF", "Loss w/ Gen"]
        results_table.field_names = ["Model", "Loss w/ Gen"]
    
        # for suffix in ["_contrastive_from_stage1_nq", "_contrastive_from_stage2_nq", "_contrastive_from_stage2_nq_lr1e3", "_contrastive_from_stage2_nq_lr1e5", "_contrastive_from_stage2_nq_lr1e5_ordered"]:
        for suffix in ["_contrastive"]:
            model_name = f"toy{suffix}"
            Path(f"output_embeddings/{train_name}").mkdir(parents=True, exist_ok=True)
            adapter_path = f"results/{train_name}/{model_name}/checkpoint_{num_map[train_name]}"
            base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
            linear_checkpoint_path = f"results/{train_name}/{model_name}/checkpoint_{num_map[train_name]}_linear.pt"
            dataset_path = f'training_datasets/autoregressive_{dataset_name}_dev_dataset_1b_contrastive_2_to_5_ctxs'
            # tf_out_embeddings_path = f'output_embeddings/{train_name}/tf_out_{dataset_name}_{suffix}_dev.npy'
            generated_embeddings_path = f'output_embeddings/{train_name}/out_{dataset_name}_{suffix}_dev.npy'
            generated_embeddings_lengths_path = f'output_embeddings/{train_name}/out_{dataset_name}_{suffix}_dev_lengths.npy'
            # generated_embeddings_lengths_path = 'aaa'
            get_split = 'dev'
            
            # print('-'*50)
            # print(f'evaluating {suffix} on {train_name} dev set')
            # outputs, loss_with_teacher_forcing, all_labels_tf = eval_with_teacher_forcing(adapter_path = adapter_path, 
            #         base_model_id = base_model_id, 
            #         dataset_path = dataset_path,
            #         linear_checkpoint_path = linear_checkpoint_path,
            #         output_path = tf_out_embeddings_path,
            #         return_full_dataset = return_full_dataset,
            #         embedding_model_dim = embedding_model_dim)
            # print('writing to ', tf_out_embeddings_path)
            # outputs_tf = np.load(tf_out_embeddings_path)
            # np.save('all_labels_tf.npy', all_labels_tf)
            # print('tf out embeddings shape', outputs_tf.shape)
            # all_labels_tf = np.load('all_labels_tf.npy')
        
            outputs, loss_with_generation, all_labels_gen = eval_with_generation(adapter_path = adapter_path,
                base_model_id = base_model_id,
                linear_checkpoint_path = linear_checkpoint_path,
                output_path = generated_embeddings_path,
                output_lengths_path = generated_embeddings_lengths_path,
                input_data_path = dataset_path, 
                get_split = get_split,
                max_new_tokens = max_new_tokens,
                embedding_model_dim = embedding_model_dim,
                compute_loss = max_new_tokens is None)
            print('writing to ', generated_embeddings_path)
            outputs_gen = np.load(generated_embeddings_path)
            
            # # np.save('all_labels_gen.npy', all_labels_gen)
            # # print('generated embeddings shape', outputs_gen.shape)
            # # all_labels_gen = np.load('all_labels_gen.npy')
            
            
            # check_hungarian_contrastive_loss(adapter_path = adapter_path, 
            #       base_model_id = base_model_id, 
            #       dataset_path = dataset_path,
            #       linear_checkpoint_path = linear_checkpoint_path,
            #       embedding_model_dim = embedding_model_dim)
            
            
            # # compare if the labels are the same
            # print(outputs_tf.shape, outputs_gen.shape)
            # for i in range(len(outputs_tf)):
            #     if not np.allclose(outputs_tf[i], outputs_gen[i], rtol=0.01, atol=0.00003):
            #         print(f"Label mismatch at index {i}")
            #         print(f"Label 1: {outputs_tf[i]}")
            #         print(f"Label 2: {outputs_gen[i]}")
            #         break
            
        #     # Add results to the table
        #     model_name = f"{train_name}{suffix}"
        #     # results_table.add_row([model_name, f"{loss_with_teacher_forcing:.4f}", f"{loss_with_generation:.4f}"])
        #     results_table.add_row([model_name, f"{loss_with_generation:.4f}"])
        
        # # Print the final results table
        # print("\nEvaluation Results:")
        # print(results_table)
    
    
    
    # # Print the final results table
    # print("\nEvaluation Results:")
    # print(results_table)
    
    # for suffix in ["_hungarian"]:
    #     adapter_path = f"results/{train_name}/toy{suffix}/checkpoint_1000"
    #     # adapter_path = None
    #     base_model_id = "meta-llama/Llama-3.2-1B-Instruct"
    #     linear_checkpoint_path = f"results/{train_name}/toy{suffix}/checkpoint_1000_linear.pt"
    #     dataset_path = f'training_datasets/autoregressive_{dataset_name}_dev_dataset_1b'
    #     tf_out_embeddings_path = f'output_embeddings/{train_name}/tf_out_{train_name}_{suffix}_dev.npy'
    #     generated_embeddings_path = f'output_embeddings/{train_name}/out_{train_name}_{suffix}_dev.npy'
    #     return_full_dataset = True
        
    #     print('-'*50)
    
    #     check_hungarian_loss(adapter_path = adapter_path,
    #         base_model_id = base_model_id,
    #         linear_checkpoint_path = linear_checkpoint_path,
    #         output_path = generated_embeddings_path,
    #         input_data_path = dataset_path, 
    #         return_full_dataset = return_full_dataset,
    #         max_new_tokens = max_new_tokens,
    #         embedding_model_dim = embedding_model_dim)
