import os

import json
import numpy as np
import random


from typing import List, Tuple
import pickle


""" 
    The script is used to check the distance between target vector embeddings.
    And also used to check the distance between predicted question embeddings. 
"""

def compute_l2_distance(query_1, query_2):
    return np.linalg.norm(query_1 - query_2)

def compute_cosine_similarity(query_1, query_2):
    return np.dot(query_1, query_2) / (np.linalg.norm(query_1) * np.linalg.norm(query_2))

def compute_averge_target_distance_same_example(target_vectors_list: List[np.ndarray]) -> Tuple[List[float], List[float]]:
    # Input -> List of Numpy arrays
    l2_distance_list = []
    cosine_similarity_list = []
    for i in range(len(target_vectors_list)):
        all_target_vectors = target_vectors_list[i]
        l2_distances = []
        cosine_similarities = []
        for j in range(len(all_target_vectors)):
            for k in range(j+1, len(all_target_vectors)):
                l2_distances.append(compute_l2_distance(all_target_vectors[j], all_target_vectors[k]))
                cosine_similarities.append(compute_cosine_similarity(all_target_vectors[j], all_target_vectors[k]))
                
        if len(l2_distances) == 0:
            print('no l2 distances', len(all_target_vectors))
            continue
        # per example avg distance
        l2_distance_list.append(np.mean(l2_distances))
        cosine_similarity_list.append(np.mean(cosine_similarities))
    return l2_distance_list, cosine_similarity_list

def normalize_np(x, p=2, dim=1, eps=1e-12):
    """
    NumPy implementation of torch.nn.functional.normalize
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    return x / norm



def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def load_model(model_name_or_path):
    if ('stella' in model_name_or_path) or ('inf-retriever' in model_name_or_path) or ('NV-Embed' in model_name_or_path):
        model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
        if 'inf-retriever' in model_name_or_path:
            model.max_seq_length = 1024
        if 'NV-Embed' in model_name_or_path:
            model.max_seq_length = 32768
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
    model.eval()
    model = model.cuda().half()
    return model


data_dir = 'data/final_eval_data/'
file_list = {"ambignq": "ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.jsonl", 
             "qampari": "dev_data_gt_qampari_corpus_5_to_8_ctxs.jsonl"}

retriever_map = {
    'stella': 'NovaSearch/stella_en_400M_v5',
    'inf': 'infly/inf-retriever-v1-1.5b',
    'contriever': 'facebook/contriever-msmarco',
    'nv-embed': 'nvidia/NV-Embed-v2',
}
retrievers = ['stella', 'inf', 'contriever', 'nv-embed']
# retrievers = ['nv-embed']
commands = ['compute', 'plot']  # ['embed', 'compute', 'plot']


# # ########################################################
# # # Embed passages
# # ########################################################
# if 'embed' in commands:
#     import torch
#     from tqdm import tqdm
#     from transformers import AutoModel, BertModel
#     import transformers
#     from sentence_transformers import SentenceTransformer

#     @torch.no_grad()
#     def embed_queries_stella(model_name_or_path, queries, model, per_gpu_batch_size=4):
#         if 'inf-retriever' in model_name_or_path:  # inf
#             query_prompt_name = "query"
#         else:                                      # stella
#             query_prompt_name = "s2p_query" 

#         model.eval()
#         embeddings, batch_question = [], []

#         for k, q in enumerate(queries):
#             batch_question.append(q)

#             if len(batch_question) == per_gpu_batch_size:
#                 embeddings.append(model.encode(batch_question, prompt_name=query_prompt_name))
#                 batch_question = []
#         if len(batch_question) > 0:
#             embeddings.append(model.encode(batch_question, prompt_name=query_prompt_name))
#         embeddings = np.concatenate(embeddings, axis=0)
#         embeddings = normalize_np(embeddings, p=2, dim=1)
#         print("Questions embeddings shape:", embeddings.shape)
#         return embeddings

#     def embed_passages_cont(passages, model, per_gpu_batch_size=4):
#         alltext = []
#         for k, p in tqdm(enumerate(passages)):
#             alltext.append(p)
        
#         with torch.no_grad():
#             allembeddings = model.encode(alltext, batch_size=per_gpu_batch_size)  # default is 512, but got oom
#         allembeddings = normalize_np(allembeddings, p=2, dim=1)
#         return allembeddings

#     def embed_passages_nv(passages, model):
#         batch_size = 4
#         with torch.no_grad():
#             def add_eos(input_examples, eos_token):
#                 input_examples = [input_example + eos_token for input_example in input_examples]
#                 return input_examples

#             allembeddings = model.encode(add_eos(passages, model.tokenizer.eos_token), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
#             allembeddings = normalize_np(allembeddings, p=2, dim=1)

#         return allembeddings
#     ####### Contriever #######
#     class Contriever(BertModel):
#         def __init__(self, config, pooling="average", **kwargs):
#             super().__init__(config, add_pooling_layer=False)
#             if not hasattr(config, "pooling"):
#                 self.config.pooling = pooling

#         def forward(
#             self,
#             input_ids=None,
#             attention_mask=None,
#             token_type_ids=None,
#             position_ids=None,
#             head_mask=None,
#             inputs_embeds=None,
#             encoder_hidden_states=None,
#             encoder_attention_mask=None,
#             output_attentions=None,
#             output_hidden_states=None,
#             normalize=False,
#         ):

#             model_output = super().forward(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 position_ids=position_ids,
#                 head_mask=head_mask,
#                 inputs_embeds=inputs_embeds,
#                 encoder_hidden_states=encoder_hidden_states,
#                 encoder_attention_mask=encoder_attention_mask,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#             )

#             last_hidden = model_output["last_hidden_state"]
#             last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

#             if self.config.pooling == "average":
#                 emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
#             elif self.config.pooling == "cls":
#                 emb = last_hidden[:, 0]

#             if normalize:
#                 emb = torch.nn.functional.normalize(emb, dim=-1)
#             return emb


#     def load_hf(object_class, model_name):
#         try:
#             obj = object_class.from_pretrained(model_name, local_files_only=True)
#         except:
#             obj = object_class.from_pretrained(model_name, local_files_only=False)
#         return obj


#     def load_retriever(model_path):
#         # try: check if model exists locally
#         path = os.path.join(model_path, "checkpoint.pth")
#         if os.path.exists(path):
#             pretrained_dict = torch.load(path, map_location="cpu")
#             opt = pretrained_dict["opt"]
#             if hasattr(opt, "retriever_model_id"):
#                 retriever_model_id = opt.retriever_model_id
#             else:
#                 # retriever_model_id = "bert-base-uncased"
#                 retriever_model_id = "bert-base-multilingual-cased"
#             tokenizer = load_hf(transformers.AutoTokenizer, retriever_model_id)
#             cfg = load_hf(transformers.AutoConfig, retriever_model_id)

#             model_class = Contriever
#             retriever = model_class(cfg)
#             pretrained_dict = pretrained_dict["model"]

#             if any("encoder_q." in key for key in pretrained_dict.keys()):  # test if model is defined with moco class
#                 pretrained_dict = {k.replace("encoder_q.", ""): v for k, v in pretrained_dict.items() if "encoder_q." in k}
#             elif any("encoder." in key for key in pretrained_dict.keys()):  # test if model is defined with inbatch class
#                 pretrained_dict = {k.replace("encoder.", ""): v for k, v in pretrained_dict.items() if "encoder." in k}
#             retriever.load_state_dict(pretrained_dict, strict=False)
#         else:
#             retriever_model_id = model_path
#             model_class = Contriever
#             cfg = load_hf(transformers.AutoConfig, model_path)
#             tokenizer = load_hf(transformers.AutoTokenizer, model_path)
#             retriever = load_hf(model_class, model_path)

#         return retriever, tokenizer, retriever_model_id

#     @torch.no_grad()
#     def embed_passages_stella(passages, model, tokenizer):
#         total = 0
#         allembeddings = []
#         batch_text = []
#         with torch.no_grad():
#             for k, p in tqdm(enumerate(passages)):
#                 batch_text.append(p)

#                 if len(batch_text) == 8 or k == len(passages) - 1:
#                     encoded_batch = tokenizer.batch_encode_plus(
#                         batch_text,
#                         return_tensors="pt",
#                         max_length=512,
#                         padding=True,
#                         truncation=True,
#                     )

#                     encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
#                     embeddings = model(**encoded_batch)

#                     embeddings = embeddings.cpu()
#                     total += len(batch_text)
#                     allembeddings.append(embeddings)

#                     batch_text = []
#                     if k % 100000 == 0 and k > 0:
#                         print(f"Encoded passages {total}")

#         allembeddings = torch.cat(allembeddings, dim=0).numpy()
#         return allembeddings


#     no_passages = 0
#     for retriever in retrievers:
#         if retriever == 'contriever':
#             model, tokenizer, _ = load_retriever(retriever_map[retriever])
#             model = model.cuda()
#         else:
#             model = load_model(retriever_map[retriever])
#         for data_name, file_name in file_list.items():
#             data = read_jsonl(os.path.join(data_dir, file_name))
#             target_vectors_list = []
#             for ex in data:
#                 gt_string = 'ground_truths' if 'ground_truths' in ex else 'positive_ctxs'
#                 if isinstance(ex[gt_string][0], list):
#                     passages = [l[0] for l in ex[gt_string]]
#                 elif isinstance(ex[gt_string][0], dict):
#                     passages = ex[gt_string]
#                 else:
#                     print(ex[gt_string][0])
#                     raise NotImplementedError
#                 passages = [p['title'] + ' ' + p['text'] for p in passages]
#                 if len(passages) == 0:
#                     print('no passages')
#                     no_passages += 1
                
#                 if retriever == 'contriever':
#                     passages_embeddings = embed_passages_cont(passages, model, tokenizer)
#                 elif retriever == 'nv-embed':
#                     passages_embeddings = embed_passages_nv(passages, model)
#                 elif retriever == 'stella' or retriever == 'inf':
#                     passages_embeddings = embed_passages_stella(passages, model)
#                 else:
#                     raise NotImplementedError
#                 target_vectors_list.append(passages_embeddings)
        
#             save_dir = f'{data_dir}/doc_embeddings/{retriever}/{file_name.replace('.jsonl', '.pkl')}'
#             os.makedirs(os.path.dirname(save_dir), exist_ok=True)
#             pickle.dump(target_vectors_list, open(save_dir, 'wb'))
            
#     print(f'no passages: {no_passages}')


########################################################
# Computing Similarity and Bin Results Based on Similarity
########################################################
if 'compute' in commands:
    retrieved_results_dir = 'results/base_retrievers/'
    retriever = 'inf'

    all_scores = []
    for data_name, file_name in file_list.items():
        scores_per_file = []
        for retriever in retrievers:
            print(f'{data_name} {retriever}')
            # first bin the examples based on the similarity
            data = read_jsonl(os.path.join(data_dir, file_name))
            if data_name == 'ambignq':
                length_list = [2,3,4,5]
                gt_string = 'positive_ctxs'
            elif data_name == 'qampari':
                length_list = [5,6,7,8]
                gt_string = 'ground_truths'
            else:
                raise NotImplementedError
            
            # get length ids
            length_ids = [[] for _ in range(len(length_list))]
            for i in range(len(data)):
                if len(data[i][gt_string]) == length_list[0]:
                    length_ids[0].append(i)
                elif len(data[i][gt_string]) == length_list[1]:
                    length_ids[1].append(i)
                elif len(data[i][gt_string]) == length_list[2]:
                    length_ids[2].append(i)
                elif len(data[i][gt_string]) == length_list[3]:
                    length_ids[3].append(i)
            # print(length_ids)
            
            # load the target_vectors_list 
            pkl_file = file_name.replace('.jsonl', '.pkl')
            print('loading ', f'{data_dir}/doc_embeddings/{retriever}/{pkl_file}')
            target_vectors_list = pickle.load(open(f'{data_dir}/doc_embeddings/{retriever}/{pkl_file}', 'rb'))
            l2_distance_list, cosine_similarity_list = compute_averge_target_distance_same_example(target_vectors_list)
            # print(l2_distance_list)
            # compute the 20, 40, 60, 80, 100 percentile of the distances
            cosine_distance_list = [1-l for l in cosine_similarity_list]
            
            for distance_list in [cosine_distance_list]:
                print('-'*100)
                # print(f'Computing Distance')
                percentile_list = []
                # k_list = [20, 40, 60, 80, 100]
                # k_list = [12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
                k_list = [25, 50, 75, 100]
                for k in k_list:
                    percentile = np.percentile(distance_list, k)
                    percentile_list.append(percentile)
                    # print(f'{k} percentile: {percentile}')

                all_bins = [[] for _ in range(len(k_list))]
                # bin the examples based on the similarity
                for i in range(len(distance_list)):
                    if distance_list[i] < percentile_list[0]:
                        all_bins[0].append(i)
                    elif distance_list[i] < percentile_list[1]:
                        all_bins[1].append(i)
                    elif distance_list[i] < percentile_list[2]:
                        all_bins[2].append(i)
                    else:
                        all_bins[-1].append(i)
                        
                score_type = 'mrecall_topk100'
                score_file = file_list[data_name].replace('.jsonl', f'_{score_type}.npy')
                scores = np.load(f'{retrieved_results_dir}/{retriever}/{score_file}')
                print('loading score from ', f'{retrieved_results_dir}/{retriever}/{score_file}')
                
                scores_inst = []
                for k, bin in zip(k_list, all_bins):
                    score_bin = np.array([scores[_id] for _id in bin])
                    print(f'Score for {score_type}, {k} percentile: {np.mean(score_bin)}')
                    scores_inst.append(np.mean(score_bin))
                scores_per_file.append(scores_inst)

                    
            for length, length_id in zip(length_list, length_ids):
                cosine_distance_list_per_length = [cosine_distance_list[i] for i in length_id]
                print(len(length_id), len(cosine_distance_list_per_length))
                for distance_list in [cosine_distance_list_per_length]:
                    print('-'*100)
                    print(f'Computing Distance for length {length} with {len(length_id)} examples')
                    print('-'*100)
                    # print(f'Computing Distance')
                    percentile_list = []
                    # k_list = [20, 40, 60, 80, 100]
                    # k_list = [12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100]
                    k_list = [25, 50, 75, 100]
                    for k in k_list:
                        percentile = np.percentile(distance_list, k)
                        percentile_list.append(percentile)
                        # print(f'{k} percentile: {percentile}')

                    all_bins = [[] for _ in range(len(k_list))]
                    # bin the examples based on the similarity
                    for i in range(len(distance_list)):
                        if distance_list[i] < percentile_list[0]:
                            all_bins[0].append(length_id[i])
                        elif distance_list[i] < percentile_list[1]:
                            all_bins[1].append(length_id[i])
                        elif distance_list[i] < percentile_list[2]:
                            all_bins[2].append(length_id[i])
                        else:
                            all_bins[-1].append(length_id[i])
                            
                    score_type = 'mrecall_topk100'
                    score_file = file_list[data_name].replace('.jsonl', f'_{score_type}.npy')
                    scores = np.load(f'{retrieved_results_dir}/{retriever}/{score_file}')
                    print('loading score from ', f'{retrieved_results_dir}/{retriever}/{score_file}')
                    
                    scores_inst = []
                    for k, bin in zip(k_list, all_bins):
                        score_bin = np.array([scores[_id] for _id in bin])
                        print(f'Score for {score_type}, {k} percentile: {np.mean(score_bin)}')
                        scores_inst.append(np.mean(score_bin))
                    scores_per_file.append(scores_inst)
        all_scores.append(scores_per_file)
            
    print(np.array(all_scores))

    # Convert to numpy array for easier manipulation
    all_scores = np.array(all_scores)
    print("Shape of all_scores:", all_scores.shape)


if 'plot' in commands:
    length_list = {
        'ambignq': [2,3,4,5],
        'qampari': [5,6,7,8]
    }
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Set up professional styling for research paper
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'lines.linewidth': 2.5,
        'patch.linewidth': 0.5,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.6,
        'ytick.minor.width': 0.6,
    })

    # Create the figure with two subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 6))

    # Define professional color palette and styling
    # Using colorblind-friendly palette
    retriever_colors = {
        'stella': '#1f77b4',  # Professional blue
        'inf': '#d62728',      # Professional red
        'contriever': '#2ca02c', # Professional green
        'nv-embed': '#9467bd' # Professional purple
    }

    # More distinct markers and line styles
    data_markers = {'ambignq': 'o', 'qampari': 's'}
    data_linestyles = {'ambignq': '-', 'qampari': '--'}
    marker_sizes = {'ambignq': 8, 'qampari': 9}

    # X-axis values (bin centers)
    x_values = [12.5, 37.5, 62.5, 87.5]

    # Dataset names with proper capitalization
    dataset_names = ['ambignq', 'qampari']
    retriever_names = ['stella', 'inf', 'contriever', 'nv-embed']
    dataset_labels = {'ambignq': 'AmbigQA', 'qampari': 'QAMPARI'}
    retriever_labels = {'stella': 'Stella', 'inf': 'Inf-retriever', 'contriever': 'Contriever', 'nv-embed': 'NV-Embed-V2'}

    # Plot for each dataset in separate subplots
    # axes = [ax1, ax2]
    for data_idx, data_name in enumerate(dataset_names):
        for length_idx in range(5):
            ax = axes[data_idx, length_idx]
            
            for ret_idx, ret_name in enumerate(retriever_names):
                scores = all_scores[data_idx, ret_idx*5 + length_idx, :]
                
                # Create professional label (just retriever name since dataset is in title)
                label = retriever_labels[ret_name]
                
                ax.plot(x_values, scores, 
                    color=retriever_colors[ret_name],
                    marker='o',  # Use same marker for both since they're on different plots
                    linestyle='-',  # Use same line style for both
                    linewidth=4.0,  # Increased from 2.5
                    markersize=12,  # Increased from 8
                    markerfacecolor=retriever_colors[ret_name],
                    markeredgecolor='white',
                    markeredgewidth=1.5,  # Increased from 1
                    label=label,
                    alpha=0.9)
            
            # Professional styling for each subplot
            # ax.set_xlabel('Embedding Distance Percentile', fontsize=20, fontweight='medium', fontfamily='Arial')
            if length_idx == 0:
                ax.set_ylabel('MRecall@100', fontsize=18, fontweight='medium', fontfamily='Arial')
            if length_idx == 0:
                ax.set_title(f'{dataset_labels[data_name]}', fontsize=23, fontweight='bold', pad=15)
            else:
                ax.set_title(f'Answer Set Size={length_list[data_name][length_idx-1]}', fontsize=23, fontweight='bold', pad=15)
            
            # Customize grid
            ax.grid(True, alpha=0.8, linestyle='-', linewidth=0.6)
            ax.set_axisbelow(True)
            
            # Set axis limits and ticks
            ax.set_xlim(0, 100)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.set_xticklabels(['(%)', '25', '50', '75', '100'], fontsize=18, fontfamily='Arial')
            
            # Set custom y-axis limits for each dataset (0.5 range each)
            if data_name == 'ambignq':
                ax.set_ylim(0.0, 1.0)
                legend_loc = 'upper right'  # Place legend at to p for AmbigNQ
            else:  # qampari
                ax.set_ylim(0.0, 0.75)
                legend_loc = 'upper right'  # Place legend at bottom for Qampari
            
            # Set y-tick label font size and family to match x-tick labels
            ax.tick_params(axis='y', labelsize=18)
            for label in ax.get_yticklabels():
                label.set_fontfamily('Arial')
            
            # Add subtle background
            ax.set_facecolor('#fafafa')
            
            # # Professional legend
            # if data_name == 'qampari':
            #     legend = ax.legend(loc=legend_loc, 
            #                     frameon=True, 
            #                     fancybox=True, 
            #                     shadow=True,
            #                     borderpad=1,
            #                     columnspacing=1.5,
            #                     handlelength=2.5,
            #                     handletextpad=0.8,
            #                     fontsize=20)
            #     legend.get_frame().set_facecolor('white')
            #     legend.get_frame().set_alpha(0.9)
            #     legend.get_frame().set_edgecolor('gray')
            #     legend.get_frame().set_linewidth(0.5)

    
    # Tight layout with padding first
    plt.tight_layout(pad=2.0)
    
    # Add single legend for the entire figure
    # Get legend handles and labels from the first subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Create a single legend for the entire figure
    fig.legend(handles, labels,
              loc='lower center',
              bbox_to_anchor=(0.5, -0.02),  # Moved closer to figure
              ncol=4,  # 4 columns for 4 retrievers
              fontsize=18,  # Slightly smaller font
              frameon=True,
              fancybox=True,
              shadow=True,
              borderpad=0.8,
              columnspacing=1.5,
              handlelength=2.0,
              handletextpad=0.6)
    
    # Add overall title
    # fig.suptitle('Retrieval Performance vs. Document Similarity Distribution', fontsize=16, fontweight='bold', y=0.98)

    # Adjust subplot spacing to make room for legend
    plt.subplots_adjust(bottom=0.17)

    # Save with high quality for research paper
    plt.savefig('figures/retrieval_performance_analysis.pdf', 
            dpi=300, 
            bbox_inches='tight', 
            facecolor='white',
            edgecolor='none',
            format='pdf')

    plt.savefig('figures/retrieval_performance_analysis.png', 
            dpi=300, 
            bbox_inches='tight', 
            facecolor='white',
            edgecolor='none',
            format='png')

    plt.show()

    print("Professional plots saved as:")
    print("  - retrieval_performance_analysis.pdf (for paper)")
    print("  - retrieval_performance_analysis.png (for preview)")

