import json
import pickle
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_clustering_results(centroids_path, labels_path):
    with open(centroids_path, 'rb') as f:
        centroids = pickle.load(f)
    with open(labels_path, 'rb') as f:
        labels = pickle.load(f)
    return centroids, labels

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_closest_document_ids(centroid, document_embeddings, labels, cluster_id, top_k=5):
    # document indices that are in the same cluster as the centroid
    doc_ids_in_cluster = [i for i, l in enumerate(labels) if l == cluster_id]
    
    # get the documents in the cluster
    document_embeddings_in_cluster = [document_embeddings[i] for i in doc_ids_in_cluster]
    
    # for each document in the cluster, calculate the cosine similarity between the document and the centroid
    similarities = [cosine_similarity(document_embedding, centroid) for document_embedding in document_embeddings_in_cluster]
    # top 5 documents with highest similarity
    max_similarity_indices = np.argsort(similarities)[::-1][:top_k]
    return [doc_ids_in_cluster[i] for i in max_similarity_indices]


def filter_cluster(closest_documents, question, system_prompt, model, tokenizer, use_vllm):
    user_prompt = "**Input Format:**\n- **User Question:** [Question]\n- **K Documents:** [Documents]".replace("[Question]", question).replace("[Documents]", '\n'.join(closest_documents))
    if use_vllm:
        outputs = model.generate(prompts=[system_prompt + "\n\n" + user_prompt])
        content = outputs[0].outputs[0].text
        thinking_content = None
    else:
        thinking_content, content = generate(system_prompt, user_prompt, tokenizer, model)
    if use_vllm:
        print(outputs)
        print(content)
    # parse the content to get the score
    score = content.split('"relevance_score":')[-1].split(',')[0].strip()
    score = int(score)
    return score, thinking_content, content

@torch.no_grad()
def generate(system_prompt, user_prompt, tokenizer, model):
    # prepare the model input
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # print("thinking content:", thinking_content) # no opening <think> tag
    # print("content:", content)
    return thinking_content, content
    
def load_model(model_name='Qwen/Qwen3-30B-A3B-Thinking-2507'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def load_vllm_model(model_name='Qwen/Qwen3-30B-A3B-Thinking-2507'):
    import os
    llm = LLM(
        model=model_name,
        max_model_len=32768,
        download_dir=os.environ['HF_HOME']
    )
    return llm
    
       
def main():
    use_vllm = False
    
    data_name = 'small'
    logger.info(f'Loading data from data/{data_name}.jsonl')
    raw_data = read_jsonl(f'data/{data_name}.jsonl')
    document_embedding_list = np.load(f'clustered_data/{data_name}_embeddings.npy')
    method = 'mean_shift'
    logger.info(f'Loading clustering results from clustered_data/{data_name}_{method}_centroids_flexible.pkl and clustered_data/{data_name}_{method}_labels_flexible.pkl')
    centroid_list, labels_list = load_clustering_results(f'clustered_data/{data_name}_{method}_centroids_flexible.pkl', f'clustered_data/{data_name}_{method}_labels_flexible.pkl')
    
    logger.info(f'Loading model from Qwen/Qwen3-30B-A3B-Thinking-2507')
    if use_vllm:
        model = load_vllm_model()
        tokenizer = None
    else:
        model, tokenizer = load_model()
    
    system_prompt = open('relevance.txt', 'r').read()
    
    logger.info(f'Filtering clusters')
    print(len(centroid_list), len(labels_list), len(document_embedding_list))
    
    all_scores = []
    outputs = []
    for i in range(len(centroid_list)):
        centroids = centroid_list[i] # (num_centroids, embedding_dim)
        labels = labels_list[i]      # (num_documents)
        document_embeddings = document_embedding_list[i]
        documents = [c['retrieval text'] for c in raw_data[i]['ctxs']]
        
        # for each centroid, find the document ids in that cluster that are closest to it
        for j, centroid in tqdm(enumerate(centroids)):
            closest_document_ids = find_closest_document_ids(centroid, document_embeddings, labels, j)
            closest_documents = [documents[i] for i in closest_document_ids]
            score, thinking_content, content = filter_cluster(closest_documents, raw_data[i]['question'], system_prompt, model, tokenizer, use_vllm)
            all_scores.append(score)
            outputs.append({"score": score, "thinking_content": thinking_content, "content": content, "question": raw_data[i]['question'], "closest_documents": closest_documents})
            
        
    print(all_scores)
    print(sum(all_scores)/float(len(all_scores)))

    with open(f'clustered_data/{data_name}_{method}_filtered_outputs.jsonl', 'w') as f:
        for output in outputs:
            f.write(json.dumps(output) + '\n')
    
    # plot the distribution of scores
    import matplotlib.pyplot as plt
    plt.hist(all_scores)
    plt.savefig('score_distribution.png')
                
if __name__ == '__main__':
    # main()
    
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507"

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    print('finished loading model')
    
    # prepare the model input
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    print('generating...')
    
    import time
    start_time = time.time()
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=16384
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("content:", content)
    print(f'Time taken: {time.time() - start_time} seconds')