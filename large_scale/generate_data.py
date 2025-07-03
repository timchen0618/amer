import argparse
import logging
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Optional
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path
import json

def read_jsonl(file_path):  
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    
    
def do_kmeans_per_instance(question_docs, num_clusters, random_state=42):
    # Store original embeddings
    original_embeddings = question_docs.copy()
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(question_docs)

    # Apply PCA
    pca = PCA(n_components=0.95)  # Retain 95% variance
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(X_pca)
    centroids_pca = kmeans.cluster_centers_
    
    # Transform centroids back to original space
    centroids_original = scaler.inverse_transform(pca.inverse_transform(centroids_pca))
    
    return cluster_labels, centroids_original, original_embeddings


def do_kmeans(filtered_data, n_clusters, random_state=42, logger=None):
    all_labels = []
    all_centroids = []
    for i in range(len(filtered_data)):
        logger.info(f"Performing K-means clustering with {n_clusters} clusters")
        labels, centroids, _ = do_kmeans_per_instance(filtered_data[i], num_clusters=n_clusters, random_state=random_state)
        all_labels.append(labels)
        all_centroids.append(centroids)
        if i > 3:
            break
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

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Data Generation Script for Large-Scale Training")
    
    # Data parameters
    parser.add_argument("--input_dir", type=str, default="data/raw",
                      help="Directory containing input data")
    parser.add_argument("--retrieval_results_dir", type=str, default="/datastor1/hungting/retrieval_outputs/mteb_retriever/",
                      help="Directory containing retrieval results")
    parser.add_argument("--output_dir", type=str, default="data/processed",
                      help="Directory to save processed data")
    parser.add_argument("--input_embeddings", type=str, default="",
                      help="Directory containing input embeddings")
    
    # Clustering parameters
    parser.add_argument("--n_clusters", type=int, default=10,
                      help="Number of clusters for K-means")
    parser.add_argument("--random_state", type=int, default=42,
                      help="Random state for reproducibility")
    
    # Processing parameters
    parser.add_argument("--batch_size", type=int, default=1000,
                      help="Batch size for processing")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of workers for parallel processing")
    
    return parser.parse_args()

# def perform_kmeans_clustering(data: np.ndarray, 
#                             n_clusters: int,
#                             random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
#     """Perform K-means clustering on the input data.
    
#     Args:
#         data: Input data array of shape (N, dim)
#         n_clusters: Number of clusters
#         random_state: Random state for reproducibility
        
#     Returns:
#         Tuple of (cluster labels, fitted KMeans model)
#     """
#     kmeans = KMeans(n_clusters=n_clusters, 
#                     random_state=random_state,
#                     n_init=10)
#     labels = kmeans.fit_predict(data)
#     return labels, kmeans

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


def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info("Starting data generation process")
    logger.info(f"Arguments: {args}")
    
    # Load data
    data = load_data(args.input_embeddings, logger)
    
    # Load retrieval results
    data_name = Path(args.input_dir).stem
    retrieval_results = load_retrieval_results(Path(args.retrieval_results_dir) / args.input_dir, logger)
    logger.info(f"Loaded retrieval results of length {len(retrieval_results)}")
    
    # # Filter data
    # filtered_data = filter_data(data, retrieval_results, logger)
    filtered_data = data
    
    # Perform clustering
    assert len(filtered_data) == len(retrieval_results)
    all_labels, all_centroids = do_kmeans(filtered_data=filtered_data, 
                                          n_clusters=args.n_clusters, 
                                          random_state=args.random_state, 
                                          logger=logger)
            
    # Save results
    save_results(all_labels, all_centroids, data_name, args.output_dir, args.n_clusters, logger)
    logger.info("Data generation process completed")
    
    
    # present results in a web visualization
    

if __name__ == "__main__":
    main()

    # python generate_data.py --input_dir inf/echo_data/t2ranking_sm.jsonl \
    #     --input_embeddings /datastor1/hungting/retrieval_outputs/mteb_retriever/inf/echo_data/t2ranking_doc_embeddings.npy \
    #     --output_dir /datastor1/hungting/clustering_results/mteb_retriever/stella-400M/