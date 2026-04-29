# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# inf_retriever.py
import os
import torch
import transformers
from training.inf_retriever.src import inbatch
import numpy as np


def load_hf(object_class, model_name, trust_remote_code=False):
    try:
        obj = object_class.from_pretrained(model_name, local_files_only=True, trust_remote_code=trust_remote_code)
    except:
        obj = object_class.from_pretrained(model_name, local_files_only=False, trust_remote_code=trust_remote_code)
    return obj


def load_retriever(model_path, pooling="last_token", random_init=False):
    # Check if model exists locally
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu", weights_only=False)
        opt = pretrained_dict["opt"]
        if hasattr(opt, "retriever_model_id"):
            retriever_model_id = opt.retriever_model_id
        else:
            # Default to INF-Retriever model
            retriever_model_id = "infly/inf-retriever-v1-1.5b"
            
        tokenizer = load_hf(transformers.AutoTokenizer, retriever_model_id, trust_remote_code=True)
        cfg = load_hf(transformers.AutoConfig, retriever_model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        cfg.pad_token_id = tokenizer.pad_token_id
        tokenizer.padding_side = "right"

        if hasattr(opt, "run_name"):
            if hasattr(opt, "training_mode"):
                print(f"Training mode: {opt.training_mode}")
                if opt.training_mode == 'standard_org_q':
                    model_class = inbatch.EmbeddingModelDocEncNoProjSingleQuery
                elif opt.training_mode == 'multi':
                    model_class = inbatch.EmbeddingModelDocEncNoProj
            else:
                raise NotImplementedError("training_mode not specified")
        else:
            raise NotImplementedError("run_name not specified")
            
        print(f"Using model class: {model_class}, retriever_model_id: {retriever_model_id}")
        pretrained_dict = pretrained_dict["model"]
        
        model = model_class(opt, None, None)
        print(f"Using model = {model_class}", flush=True)
        model.load_state_dict(pretrained_dict, strict=True)
        model.eval()
        print('Finished loading model')
        if opt.training_mode == 'standard_org_q':
            retriever = model.encoder
        else:
            retriever = model
        retriever._is_multi_query = (opt.training_mode != 'standard_org_q')


    else:
        raise NotImplementedError("model_path not specified. If you are loading a pretrained model, do not use this function.")

    return retriever, tokenizer, retriever_model_id



def normalize_np(x, p=2, dim=1, eps=1e-12):
    """
    NumPy implementation of torch.nn.functional.normalize
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    return x / norm


@torch.no_grad()
def embed_queries(args, queries, model):
    def add_eos(input_examples, eos_token):
        input_examples = [input_example + eos_token for input_example in input_examples]
        return input_examples

    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",
                             "example_input": "Given a question and some relevant passages, retrieve passages that answer the question but are not in the input set.",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
    model.eval()
    embeddings, batch_question = [], []
    batch_size = args.per_gpu_batch_size
    for k, q in enumerate(queries):
        if args.lowercase:
            q = q.lower()
        if args.normalize_text:
            q = src.normalize_text.normalize(q)
        batch_question.append(q)
        # get the embeddings
        max_length = 32768
        
        if len(batch_question) == batch_size:
            embedding = model.encode(batch_question, instruction=query_prefix, max_length=max_length).cpu().numpy()
            embeddings.append(embedding)
            batch_question = []
    if len(batch_question) > 0:
        embedding = model.encode(batch_question, instruction=query_prefix, max_length=max_length).cpu().numpy()
        embeddings.append(embedding)
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = normalize_np(embeddings, p=2, dim=1)
    print("Questions embeddings shape:", embeddings.shape)
    return embeddings

@torch.no_grad()
def embed_queries_stella(args, queries, model):
    if ('inf-retriever' in args.model_name_or_path) or ('Qwen' in args.model_name_or_path):
        query_prompt_name = "query"
    else:
        query_prompt_name = "s2p_query"

    model.eval()
    embeddings, batch_question = [], []

    for k, q in enumerate(queries):
        if args.lowercase:
            q = q.lower()
        if args.normalize_text:
            q = src.normalize_text.normalize(q)
        batch_question.append(q)

        if len(batch_question) == args.per_gpu_batch_size:
            embeddings.append(model.encode(batch_question, prompt_name=query_prompt_name))
            batch_question = []
    if len(batch_question) > 0:
        embeddings.append(model.encode(batch_question, prompt_name=query_prompt_name))
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = normalize_np(embeddings, p=2, dim=1)
    print("Questions embeddings shape:", embeddings.shape)
    return embeddings


@torch.no_grad()
def embed_queries_iterative_retrieval(args, queries, model):
    def last_token_pool(last_hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    model.eval()
    embeddings, batch_question = [], []
    max_length = 1024
    if 'infly' in args.model_name_or_path:
        tokenizer_path = 'infly/inf-retriever-v1-1.5b'
    elif 'qwen3' in args.model_name_or_path:
        tokenizer_path = 'Qwen/Qwen3-Embedding-0.6B'
    elif 'contriever' in args.model_name_or_path:
        tokenizer_path = 'facebook/contriever-msmarco'
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)
        

    for k, q in enumerate(queries):
        if args.lowercase:
            q = q.lower()
        if args.normalize_text:
            q = src.normalize_text.normalize(q)
        batch_question.append(get_detailed_instruct(task, q))

        if len(batch_question) == args.per_gpu_batch_size:
            batch_dict = tokenizer(batch_question, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            for k, v in batch_dict.items():
                batch_dict[k] = v.to(model.device)
            outputs = model(**batch_dict)
            docs_vectors = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().numpy()

            embeddings.append(docs_vectors)
            batch_question = []
    if len(batch_question) > 0:
        batch_dict = tokenizer(batch_question, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        for k, v in batch_dict.items():
            batch_dict[k] = v.to(model.device)
        outputs = model(**batch_dict)
        docs_vectors = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask']).cpu().numpy()

        embeddings.append(docs_vectors)
        
    embeddings = np.concatenate(embeddings, axis=0)
    embeddings = normalize_np(embeddings, p=2, dim=1)
    print("Questions embeddings shape:", embeddings.shape)
    return embeddings