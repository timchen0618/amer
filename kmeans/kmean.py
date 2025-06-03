import numpy as np
import json

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import json
import regex
import string
import unicodedata

from tqdm import trange, tqdm

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

        
        
def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data


def do_kmeans(question_docs, num_clusters):
    # Store original embeddings
    original_embeddings = question_docs.copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(question_docs)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Retain 95% variance
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_pca)
    centroids_pca = kmeans.cluster_centers_
    
    # Transform centroids back to original space
    centroids_original = scaler.inverse_transform(pca.inverse_transform(centroids_pca))
    
    return cluster_labels, centroids_original, original_embeddings

def generate_web_data(doc_embeddings, data, num_clusters):
    results = []
    for q_idx in range(len(doc_embeddings)):
        inst = data[q_idx]
        
        cluster_labels, centroids, original_embeddings = do_kmeans(doc_embeddings[q_idx], num_clusters)
        
        question_clusters = []
        for c_idx, centroid in enumerate(centroids):
            # Get all documents in this cluster
            cluster_indices = np.where(cluster_labels == c_idx)[0]
            cluster_size = len(cluster_indices)
            
            # Sort documents by their original retriever ranking (lower index = higher rank)
            top_3_indices = sorted(cluster_indices)[:3]
            
            top_docs = []
            for doc_idx in top_3_indices:
                top_docs.append({
                    'text': inst['ctxs'][doc_idx]['text'],
                    'ranking': int(doc_idx),
                    'is_gold': inst['ctxs'][doc_idx].get('is_gold', False)
                })
            
            question_clusters.append({
                'size': int(cluster_size),
                'centroid': centroid.tolist(),
                'topDocs': top_docs
            })
        
        results.append({
            'question': inst['question'],
            'clusters': question_clusters
        })
    
    return results

def main(doc_embeddings, data, num_clusters):
    num_questions = doc_embeddings.shape[0]
    out_data = []
    out_centroids = []
    out_lens = []
    for q_idx in trange(num_questions):
        # Get embeddings for current question
        inst = data[q_idx]
        
        cluster_labels, centroids, original_embeddings = do_kmeans(doc_embeddings[q_idx], num_clusters)
        out_centroids.append(centroids)
        out_lens.append(len(centroids))
        out_clusters = []
        for c_idx, centroid in enumerate(centroids):
            # Get all documents in this cluster
            cluster_indices = np.where(cluster_labels == c_idx)[0]
            cluster_size = len(cluster_indices)
            
            # Sort documents by their original retriever ranking (lower index = higher rank)
            top_indices = sorted(cluster_indices)[:1]
            
            # Sort documents by their cosine similarity to the centroid in original space
            closest_indices = sorted(cluster_indices, 
                                   key=lambda x: cosine_similarity([centroid], [original_embeddings[x]])[0])[:3]
            
            out_clusters.append({
                "size": cluster_size,
                "top_ranked_doc": inst['ctxs'][top_indices[0]],
                "closest_docs": [inst['ctxs'][doc_idx] for doc_idx in closest_indices],
            })
            
        out_data.append({
            "question": inst['question'],
            "clusters": out_clusters
        })
        
    print(np.concatenate(out_centroids, axis=0).shape)
    print(np.array(out_lens).shape)
    return out_data, np.concatenate(out_centroids, axis=0), np.array(out_lens)
            
            # print(f"\nCluster {c_idx + 1}:")
            # print(f"Number of documents in cluster: {cluster_size}")
            # print(f"Centroid embedding: {centroid}")
            # print("Top 3 documents by original retriever ranking:")
            # for rank, doc_idx in enumerate(top_3_indices, 1):
            #     print(f"  {rank}. Document index: {doc_idx}")
            #     print(f"  Document: {inst['ctxs'][doc_idx]['text']}")
            #     if 'is_gold' in inst['ctxs'][doc_idx]:
            #         print(f"  Is gold answer: {inst['ctxs'][doc_idx]['is_gold']}")

    """                         
        For the final data, we need to have the following:
        - question
        - clusters
            - the top 3 docs in the cluster, and the top-ranked doc in the cluster
            - the cluster centroid
            - generate a description / summary of the cluster using GPT-4
            - Later we will filter out irrelevant clusters based on the description, using GPT-4
            
        So we will save 2 things:
        1. question mapping to {"description": cluster centroids}
        2. data, where each instance is a question, and the top 3 docs in the cluster
        {
            "question": question,
            "clusters": [
                {
                    "size": cluster_size,
                    "top_ranked_doc": top_ranked_doc,
                    "closest_docs": closest_docs,
                    "description": description
                }
            ]
        }
            
    """

def annotate_retrieval_outputs(data, qampari_data):
    assert len(data) == len(qampari_data)
    tok = SimpleTokenizer()
    
    for inst, qampari_inst in tqdm(zip(data, qampari_data)):
        assert inst['question'].strip('\n') == qampari_inst['question_text'], (inst['question'], qampari_inst['question_text'])
        
        for answer in qampari_inst['answer_list']:
            for doc in inst['ctxs']:
                is_gold = has_answer(answer['aliases'], doc['text'], tok)
                doc['is_gold'] = is_gold
                # if is_gold:
                #     print(f"Question: {inst['question']}")
                #     print(f"Answer: {answer['aliases']}")
                #     print(f"Document: {doc['text']}")
                #     print("-" * 100)
    return data
                    

if __name__ == "__main__":
    # doc_embeddings is a 3D numpy array of shape (num_question, num_docs, embedding_dim)
    # for each question, perform k-means clustering among the document embeddings, and print out the centroid of the cluster
    # not only print out the embedding, but also print out the question and the document that is closest to the centroid
    qampari_path = "/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/train_data.jsonl"
    data_path = "/scratch/cluster/hungting/projects/Multi_Answer/mteb_retriever/outputs/inf/all_train_questions_q_sm.json"
    doc_embeddings_path = "doc_embeddings_q_sm_500.npy"
    
    
    qampari_data = read_jsonl(qampari_path)[:500]
    doc_embeddings = np.load(doc_embeddings_path)
    data = read_jsonl(data_path)[:500]
    num_clusters = 10  # You can adjust this number
    
    # data = annotate_retrieval_outputs(data, qampari_data)
    print('start doing clustering...')
    out_data, out_centroids, out_lens = main(doc_embeddings=doc_embeddings, 
         data=data, num_clusters=num_clusters, 
         )
    
    with open("cluster_results_qampari_q_sm_500.json", "w") as f:
        json.dump(out_data, f, indent=4)
    
    np.save("../data_creation/raw_data/qampari_q_sm_500/out_centroids_qampari_q_sm_500.npy", out_centroids)
    np.save("../data_creation/raw_data/qampari_q_sm_500/out_lens_qampari_q_sm_500.npy", out_lens)
    
    
    # # Generate data for web visualization
    # web_data = generate_web_data(doc_embeddings, data, num_clusters)
    
    # output_file="cluster_results_qampari.json"
    # # Save the data to a JSON file
    # with open(output_file, 'w') as f:
    #     json.dump(web_data, f, indent=2)
    
    # print("Results have been saved to cluster_results.json")


    


