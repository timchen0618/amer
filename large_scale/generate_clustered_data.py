import argparse
import logging
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from typing import Tuple, List, Optional
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import json
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
import numpy as np
from tqdm import trange
import json
import pickle
import random
import csv
import sys

csv.field_size_limit(sys.maxsize)


def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def write_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def read_corpus(file_path):
    import csv
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if row[0] == "id":
                continue
            data.append(row)
    return data


def read_jsonl(file_path):  
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
   
def load_model(model_name: str):
    if ('stella' in model_name) or ('inf-retriever' in model_name):
        model = SentenceTransformer(model_name, trust_remote_code=True)
        if 'inf-retriever' in model_name:
            model.max_seq_length = 8192
    elif ('nv' in model_name):
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
        
    print('finish loading model')
    
    model.eval()
    model = model.cuda()
    return model

def normalize_np(x, p=2, dim=1, eps=1e-12):
    """
    NumPy implementation of torch.nn.functional.normalize
    """
    norm = np.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = np.maximum(norm, eps)  # Avoid division by zero
    return x / norm

@torch.no_grad()
def embed_passages_stella(passages, model):
    batch_size = 8
    batch_texts = []
    allembeddings = []
    
    # Embed the documents
    for row in passages:
        if 'title' in row:
            batch_texts.append(str(row['title']) + ' ' + str(row['text']))
        else:
            if 'text' in row:
                batch_texts.append(str(row['text']))
            else:
                batch_texts.append(str(row['retrieval text']))
        # if len(batch_texts) == batch_size:
        #     docs_vectors = model.encode(batch_texts)
        #     # add embeddings and ids
        #     allembeddings.append(docs_vectors)
        #     batch_texts = []
    # process the last batch
    # if len(batch_texts) > 0:
    #     docs_vectors = model.encode(batch_texts)
    #     allembeddings.append(docs_vectors)
    allembeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=True)
    # allembeddings = np.concatenate(allembeddings, axis=0)
    allembeddings = normalize_np(allembeddings, p=2, dim=1)
    return allembeddings    

def do_pca(original_embeddings):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(original_embeddings)
    pca = PCA(n_components=0.95)  # Retain 95% variance
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, scaler, pca


def do_kmeans_per_instance(question_docs, num_clusters, random_state=42):
    # Store original embeddings
    original_embeddings = question_docs.copy()
    X_pca, scaler, pca = do_pca(original_embeddings)

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_pca)
    centroids_pca = kmeans.cluster_centers_
    
    # Transform centroids back to original space
    centroids_original = scaler.inverse_transform(pca.inverse_transform(centroids_pca))
    
    return cluster_labels, centroids_original, original_embeddings


def do_dbscan_per_instance(original_embeddings, eps=2, min_samples=2):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Store original embeddings
    X_pca, scaler, pca = do_pca(original_embeddings)
    
    cluster_labels = dbscan.fit_predict(X_pca)
    core_point_indices = dbscan.core_sample_indices_
    
    centroids = []
    # 3. Initialize a dictionary to store core points for each cluster
    core_points_by_cluster = {}

    # 4. Iterate through unique cluster labels (excluding noise, which is -1)
    unique_labels = set(cluster_labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    unique_labels = sorted(list(unique_labels))

    for label in unique_labels:
        # Find the indices of data points belonging to the current cluster
        cluster_indices = np.where(cluster_labels == label)[0]

        # Find the core points that are also in this cluster
        # This is done by finding the intersection of the two index sets
        core_points_in_cluster_indices = np.intersect1d(core_point_indices, cluster_indices)

        # Get the actual data points
        core_points_in_cluster = X_pca[core_points_in_cluster_indices]
        
        # Store them in the dictionary
        core_points_by_cluster[label] = core_points_in_cluster
    
        # Calculate the centroid of core points in the current cluster
        core_centroid = np.mean(core_points_in_cluster, axis=0)

        # Find the core point closest to this centroid
        distances = np.linalg.norm(core_points_in_cluster - core_centroid, axis=1)
        most_central_index = np.argmin(distances)
        representative_core_point = core_points_in_cluster[most_central_index]
        centroids.append(representative_core_point.reshape(1, -1))
    
    if len(centroids) == 0:
        # reduce to core points
        centroids_original = original_embeddings[core_point_indices]
    else:
        centroids_pca = np.concatenate(centroids, axis=0)
        
        # Transform centroids back to original space
        centroids_original = scaler.inverse_transform(pca.inverse_transform(centroids_pca))
    
    return cluster_labels, centroids_original


def do_mean_shift_per_instance(original_embeddings):
    X_pca, scaler, pca = do_pca(original_embeddings)
    ms = MeanShift()
    labels = ms.fit_predict(X_pca)
    centroids_pca = ms.cluster_centers_
    centroids_original = scaler.inverse_transform(pca.inverse_transform(centroids_pca))
    return labels, centroids_original


def do_clustering(filtered_data, n_clusters, eps, min_samples, random_state=42, logger=None, method='kmeans'):
    all_labels = []
    all_centroids = []
    if method == 'kmeans':
        logger.info(f"Performing K-means clustering with {n_clusters} clusters")
    elif method == 'dbscan':
        logger.info(f"Performing DBSCAN clustering with eps={eps} and min_samples={min_samples}")
    elif method == 'mean_shift':
        logger.info(f"Performing Mean Shift clustering")
        
    for i in trange(len(filtered_data)):
        if method == 'kmeans':
            labels, centroids, _ = do_kmeans_per_instance(filtered_data[i], num_clusters=n_clusters, random_state=random_state)
        elif method == 'dbscan':
            labels, centroids = do_dbscan_per_instance(filtered_data[i], eps=eps, min_samples=min_samples)
        elif method == 'mean_shift':
            labels, centroids = do_mean_shift_per_instance(filtered_data[i])
        else:
            raise ValueError(f"Unsupported method: {method}")
        all_labels.append(labels)
        all_centroids.append(centroids)
    # print('all_labels', all_labels)
    # print('all_centroids', all_centroids)
    return all_labels, all_centroids


def setup_logger(log_dir: str = "logs") -> logging.Logger:
    """Setup logger for data generation process.
    
    Args:
        log_dir: Directory to store log files
        
    Returns:
        Logger object configured for data generation
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"data_generation_{timestamp}.log")
    
    logger = logging.getLogger("data_generation")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def load_data(input_dir: str, logger: logging.Logger) -> np.ndarray:
    """Load data from the input directory.
    
    Args:
        input_dir: Directory containing input data
        logger: Logger object
        
    Returns:
        Loaded data as numpy array
    """
    data = np.load(input_dir)
    logger.info(f"Loaded data from {input_dir} with shape: {data.shape}")
    return data

def load_retrieval_results(input_dir: str, logger: logging.Logger) -> List[dict]:
    """Load retrieval results from the input directory.
    
    Args:
        input_dir: Directory containing retrieval results
        logger: Logger object
        
    Returns:
        List of retrieval results
    """
    logger.info(f"Loading retrieval results from {input_dir}")
    return read_jsonl(input_dir)
    # if str(input_dir).endswith(".jsonl"):
    #     return read_jsonl(input_dir)
    # elif str(input_dir).endswith(".json"):
    #     return read_json(input_dir)
    # else:
    #     raise ValueError(f"Unsupported file type: {input_dir}")

def filter_data(data: np.ndarray, 
                retrieval_results: List[dict],
                logger: logging.Logger) -> np.ndarray:
    """Filter data based on retrieval results.
    
    Args:
        data: Input data array
        retrieval_results: List of retrieval results
        logger: Logger object
        
    Returns:
        Filtered data array
    """
    logger.info("Filtering data based on retrieval results")
    # TODO: Implement actual filtering logic
    return data

def save_results(all_labels, all_centroids, data_name, output_dir, n_clusters, logger):
    all_labels = np.array(all_labels)
    print('all_labels.shape', all_labels.shape)
    all_centroids = np.array(all_centroids)
    print('all_centroids.shape', all_centroids.shape)
    # Save results
    output_file = Path(output_dir) / f'{data_name}_kmeans_{n_clusters}_labels.npy'
    np.save(output_file, all_labels)
    logger.info(f"Saved clustering labels to {output_file}")
    
    output_file = Path(output_dir) / f'{data_name}_kmeans_{n_clusters}_centroids.npy'
    np.save(output_file, all_centroids)
    logger.info(f"Saved clustering centroids to {output_file}")


def save_clustering_results_flexible(labels, centroids, embeddings, data_name, output_dir, method, n_clusters, eps, min_samples):
    """Save clustering results in flexible formats that handle variable-length data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save as pickle files (most flexible for Python objects)
    with open(output_dir / f'{data_name}_{method}_labels_flexible.pkl', 'wb') as f:
        pickle.dump(labels, f)
    
    with open(output_dir / f'{data_name}_{method}_centroids_flexible.pkl', 'wb') as f:
        pickle.dump(centroids, f)
            
    # Also save metadata as JSON
    metadata = {
        'num_questions': len(labels),
        'cluster_counts': [len(np.unique(label_set)) for label_set in labels] if isinstance(labels, list) else 'uniform',
        'doc_counts': [len(emb) for emb in embeddings] if isinstance(embeddings, list) else 'uniform',
        'format': 'flexible_lists',
        'n_clusters': n_clusters,
        'eps': eps,
        'min_samples': min_samples,
        'method': method
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved flexible clustering results to {output_dir}")
    print(f"Metadata: {metadata}")
    
    return output_dir
 




def create_negatives(question_type):
    #########################################################
    # # create negatives
    #########################################################
    # corpus = read_corpus('/scratch/hc3337/MassiveDS-140B/massive_ds_140b.tsv')
    # print('finished reading corpus')
    # for question_type in ["eli5", "researchy_questions", "ner_retrieve"]:
    #     data = read_json(f'/scratch/hc3337/projects/Multi_Answer/mteb_retriever/data/{question_type}.json')
    #     # write_jsonl(data, f'/scratch/hc3337/projects/autoregressive/data_creation/raw_data/{question_type}_train_question_only.jsonl')
        
    #     len_corpus = len(corpus)
    #     out_data = []
    #     for _ in range(len(data)):
    #         random_idx = random.randint(0, len_corpus - 1)
            
    #         out_data.append({"id": corpus[random_idx][0], "text": corpus[random_idx][1], "title": corpus[random_idx][2] if len(corpus[random_idx]) > 2 else ""})
        
    #     write_jsonl(out_data, f'/scratch/hc3337/projects/autoregressive/data_creation/raw_data/{question_type}_train_random_negatives.jsonl')
    
    # #########################################################
    # # # create embeddings for negatives
    # #########################################################
    # model = load_model('infly/inf-retriever-v1-1.5b')    
    # # for question_type in ["eli5", "researchy_questions", "ner_retrieve"]:
    # data = read_jsonl(f'/scratch/hc3337/projects/autoregressive/data_creation/raw_data/{question_type}_train_random_negatives.jsonl')
    # embeddings = embed_passages_stella(data, model)
    # os.makedirs(f'/scratch/hc3337/projects/autoregressive/data_creation/raw_data/{question_type}_inf', exist_ok=True)
    # np.save(f'/scratch/hc3337/projects/autoregressive/data_creation/raw_data/{question_type}_inf/{question_type}_train_random_embeddings_10.npy', embeddings)
        
    #########################################################
    # # create embeddings for positives
    #########################################################
    rootdir = f'/scratch/hc3337/projects/autoregressive/large_scale/clustered_data/{question_type}/'
    centroid_file = np.load(f'{rootdir}/{question_type}_inf+contriever_top500_split_1_kmeans_10_labels.npy')
    print('centroid_file.shape', centroid_file.shape)
        
        
def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info("Starting data generation process")
    logger.info(f"Arguments: {args}")
    import time
    start_time = time.time()
    # # Load data
    # data = load_data(args.input_embeddings, logger)
    
    # Load retrieval results
    data_name = Path(args.input_file).stem
    retrieval_results = load_retrieval_results(Path(args.retrieval_results_dir) / args.input_file, logger)
    logger.info(f"Loaded retrieval results of length {len(retrieval_results)}")
    
    # genereate embeddings
    if args.input_embeddings != "":
        data = np.load(args.input_embeddings)
        print('loaded embeddings from ', args.input_embeddings, 'with shape', data.shape)
    else:
        model = load_model(args.model_name)
        data = []
        for inst in retrieval_results:
            embeddings = embed_passages_stella(inst['ctxs'], model)
            data.append(embeddings)
    
    # # Filter data
    # filtered_data = filter_data(data, retrieval_results, logger)
    filtered_data = data
    
    # Perform clustering
    assert len(filtered_data) == len(retrieval_results)
    all_labels, all_centroids = do_clustering(filtered_data=filtered_data, 
                                          n_clusters=args.n_clusters, 
                                          eps=args.eps,
                                          min_samples=args.min_samples,
                                          random_state=args.random_state, 
                                          logger=logger,
                                          method=args.method)
            
    # Save results
    # save_results(all_labels, all_centroids, data_name, args.output_dir, args.n_clusters, logger)
    save_clustering_results_flexible(all_labels, all_centroids, data, data_name, args.output_dir, args.method, args.n_clusters, args.eps, args.min_samples)
    
    if args.save_embeddings:
        data = [l.reshape(1, -1, l.shape[-1]) for l in data]
        data = np.concatenate(data, axis=0)
        np.save(f'{args.output_dir}/{data_name}_embeddings.npy', data)
        print('saved embeddings to ', f'{args.output_dir}/{data_name}_embeddings.npy')
    logger.info("Data generation process completed")
    end_time = time.time()
    logger.info(f"Time taken: {end_time - start_time} seconds")
    
    # present results in a web visualization
    
def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Data Generation Script for Large-Scale Training")
    
    parser.add_argument("--model_name", type=str, default="stella-400M",
                      help="Model name")
    # Data parameters
    parser.add_argument("--input_file", type=str, default="data/raw",
                      help="Directory containing input data")
    parser.add_argument("--retrieval_results_dir", type=str, default="/datastor1/hungting/retrieval_outputs/mteb_retriever/",
                      help="Directory containing retrieval results")
    parser.add_argument("--output_dir", type=str, default="/datastor1/hungting/retrieval_outputs/mteb_retriever/",
                      help="Directory to save processed data")
    parser.add_argument("--save_embeddings", action="store_true", default=False,
                      help="Save embeddings")
    parser.add_argument("--method", type=str, default="dbscan",
                      help="Clustering method")
    parser.add_argument("--input_embeddings", type=str, default="",
                      help="Path to input embeddings")
    
    # Clustering parameters
    parser.add_argument("--n_clusters", type=int, default=10,
                      help="Number of clusters for K-means")
    parser.add_argument("--eps", type=float, default=2,
                      help="Epsilon for DBSCAN")
    parser.add_argument("--min_samples", type=int, default=3,
                      help="Minimum samples for DBSCAN")
    parser.add_argument("--random_state", type=int, default=42,
                      help="Random state for reproducibility")
    
    # Processing parameters
    parser.add_argument("--batch_size", type=int, default=1000,
                      help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of workers for parallel processing")
    
    return parser.parse_args()

if __name__ == "__main__":
    main()

    # create_negatives(sys.argv[1])




    ### Command ###
    # python large_scale/generate_data.py --model_name infly/inf-retriever-v1-1.5b --input_file eli5_inf+contriever_top500_split_0.jsonl --retrieval_results_dir /scratch/hc3337/projects/autoregressive/large_scale/data/eli5/ --output_dir /scratch/hc3337/projects/autoregressive/large_scale/clustered_data/eli5/
    
    # cd large_scale/
    # python generate_clustered_data.py --model_name infly/inf-retriever-v1-1.5b --input_file small.jsonl --retrieval_results_dir data/ --output_dir clustered_data/ --input_embeddings clustered_data/small_embeddings.npy --method dbscan --eps 2 --min_samples 3
    # python generate_clustered_data.py --model_name infly/inf-retriever-v1-1.5b --input_file small.jsonl --retrieval_results_dir data/ --output_dir clustered_data/ --input_embeddings clustered_data/small_embeddings.npy --method kmeans --n_clusters 10
    # python generate_clustered_data.py --model_name infly/inf-retriever-v1-1.5b --input_file small.jsonl --retrieval_results_dir data/ --output_dir clustered_data/ --input_embeddings clustered_data/small_embeddings.npy --method mean_shift
    
    
    
    ### visualize clusters ###
    # python visualize_clusters.py         --labels_path clustered_data/small_dbscan_labels_flexible.pkl --centroids_path  clustered_data/small_dbscan_centroids_flexible.pkl   --embeddings_path  clustered_data//small_embeddings.npy  --output_path clustering_visualization_dbscan.html     --retrieved_documents_path  /scratch/hc3337/projects/autoregressive/large_scale/data/small.jsonl
    
    # python visualize_clusters.py         --labels_path clustered_data/small_mean_shift_labels_flexible.pkl --centroids_path  clustered_data/small_mean_shift_centroids_flexible.pkl   --embeddings_path  clustered_data//small_embeddings.npy  --output_path clustering_visualization_mean_shift.html     --retrieved_documents_path  /scratch/hc3337/projects/autoregressive/large_scale/data/small.jsonl