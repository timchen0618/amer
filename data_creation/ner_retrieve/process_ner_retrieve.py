import json
from pathlib import Path
from tqdm import tqdm
import logging
import concurrent.futures
from typing import List, Dict, Any
import os
import time
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Read JSONL file
    Args:
        file_path: Path to the JSONL file
    """
    logger.info(f"Reading JSONL file from: {file_path}")
    time_start = time.time()
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line))
    
    logger.info(f"Successfully loaded {len(results)} records in {time.time() - time_start} seconds")
    return results

"""
Data format:
{
    "id": "XAZbWYcB1INCf0Uy5i5T", 
    "doc_id": "International Laser Display Association-3", 
    "title": "International Laser Display Association", 
    "page_id": "2142294", "para_index": 3, 
    "tagged_entities": {
        entity_type_1: {
            "entity_1": {"entity_mention": [entity_mention_1, entity_mention_2, ...]}, 
            "entity_2": {"entity_mention": [entity_mention_1, entity_mention_2, ...]}, 
            ...
        }, 
        entity_type_2: {
            "entity_1": {"entity_mention": [entity_mention_1, entity_mention_2, ...]}, 
            "entity_2": {"entity_mention": [entity_mention_1, entity_mention_2, ...]}, 
            ...
        }, 
    },
    "tagged_entity_types": [entity_type_1, entity_type_2, ...]
}

"""

def get_entity_type_list(data):
    logger.info("Starting to process NER retrieve data")
    entity_type_list = set()
    for item in tqdm(data):
        for entity_type in item["tagged_entities"]:
            entity_type_list.add(entity_type)
    logger.info(f"Found {len(entity_type_list)} unique entity types")
    return list(entity_type_list)


def get_all_entity_list(data):
    entity_list = set()
    for item in tqdm(data):
        for entity_type in item["tagged_entities"]:
            for entity in item["tagged_entities"][entity_type].keys():
                entity_list.add(entity)
    return list(entity_list)


def plot_histogram(data, entity_type_list, num_samples=2, threshold=100):
    """
        First, we have to get the gold documents of each entity type. 
        Also keep the mapping of each entity type to a list of entities. 
        Then we get two / three possible entity types, and we check the intersection of the gold documents. 
        Plot the histogram of the number of documents in the intersection. Only keep the document ids in the intersection. 
    """    
    entity_type_to_gold_docs = {}
    for entity_type in entity_type_list:
        entity_type_to_gold_docs[entity_type] = set()
    for item in data:
        for entity_type in item["tagged_entities"]:
            entity_type_to_gold_docs[entity_type].add(item['doc_id'])
            
    logger.info(f"Finished getting gold documents for each entity type")
    
    logger.info(f"Starting to get the intersection of gold documents for each pair of entity types")
    num_docs_counter = Counter()
    query_list = []
    if num_samples == 2:
        for entity_type_1, entity_type_2 in tqdm(combinations(entity_type_list, num_samples)):
            intersection = entity_type_to_gold_docs[entity_type_1] & entity_type_to_gold_docs[entity_type_2]
            num_docs_counter[len(intersection)] += 1
            if len(intersection) >= threshold:
                query_list.append("Retrieve all mentions about {} AND {}".format(entity_type_1, entity_type_2))
    elif num_samples == 3:
        for entity_type_1, entity_type_2, entity_type_3 in tqdm(combinations(entity_type_list, num_samples)):
            intersection = entity_type_to_gold_docs[entity_type_1] & entity_type_to_gold_docs[entity_type_2] & entity_type_to_gold_docs[entity_type_3]
            num_docs_counter[len(intersection)] += 1
            if len(intersection) >= threshold:
                query_list.append("Retrieve all mentions about {} AND {} AND {}".format(entity_type_1, entity_type_2, entity_type_3))
    logger.info(f"Finished getting the intersection of gold documents for each pair of entity types")
    
    # save the counter to a json file
    with open(f"num_docs_counter_{num_samples}.json", "w") as f:
        json.dump(num_docs_counter, f)
        
    print(num_docs_counter)
    return query_list
    
    # # plot the histogram
    # plt.figure(figsize=(12, 6))
    # plt.bar(num_docs_counter.keys(), num_docs_counter.values(), color='skyblue')
    # plt.xlabel('Number of Documents in Intersection')
    # plt.ylabel('Frequency')
    # plt.title(f'Distribution of Document Intersections between {num_samples} Entity Types')
    # plt.grid(True, alpha=0.3)
    
    # # Add value labels on top of each bar
    # for x, y in zip(num_docs_counter.keys(), num_docs_counter.values()):
    #     plt.text(x, y, str(y), ha='center', va='bottom')
    
    # # Adjust layout to prevent label cutoff
    # plt.tight_layout()
    
    # # Save the histogram
    # plt.savefig(f"histogram_{num_samples}.png", dpi=300, bbox_inches='tight')
    # plt.close()  # Close the figure to free memory
    
        
    
def process_corpus(corpus_path):
    corpus = read_jsonl(corpus_path)
    unique_doc_ids = set()
    unique_ids = set()
    for item in tqdm(corpus):
        unique_doc_ids.add(item["doc_id"])
        unique_ids.add(item["id"])
    logger.info(f"Found {len(unique_doc_ids)} unique document ids")
    logger.info(f"Found {len(unique_ids)} unique ids")
    return unique_doc_ids, unique_ids
    




if __name__ == "__main__":
    root_dir = Path("/scratch/cluster/hungting/projects/autoregressive/data/NERetrieve/Retrieval")
    list_root_dir = Path("/scratch/cluster/hungting/projects/autoregressive/data/NERetrieve/entity_lists")
    split = "train"
    command = "get_all_entity_list"
    
    
    
    if command == "get_entity_type_list":
        data = read_jsonl(root_dir / f"NERetrive_IR_{split}.jsonl")
        logger.info("Start getting entity type list")
        entity_type_list = get_entity_type_list(data)
        logger.info("Finished getting entity type list")
    
        
        fw = open(list_root_dir / f"entity_type_list_{split}.txt", "w")
        for entity_type in entity_type_list:
            fw.write(entity_type + "\n")
        fw.close()
        logger.info("Entity type list saved to: {}".format(list_root_dir / f"entity_type_list_{split}.txt"))
        
    elif command == "get_all_entity_list":
        # data = []
        # for split in ["train", "test"]:
        #     data.extend(read_jsonl(root_dir / f"NERetrive_IR_{split}.jsonl"))
            
        # logger.info("Start getting all entity list")
        # all_entity_list = get_all_entity_list(data)
        # logger.info("Finished getting all entity list")
        
        # fw = open(list_root_dir / f"all_entity_list.txt", "w")
        # for entity in all_entity_list:
        #     fw.write(entity + "\n")
        # fw.close()
        # logger.info("All entity list saved to: {}".format(list_root_dir / f"all_entity_list.txt"))
        all_entity_list = [line.strip('\n').strip() for line in open(list_root_dir / f"all_entity_list.txt")]
        query_list = ["Give me information about {}".format(l) for l in all_entity_list if len(l) > 0]
        fw = open(list_root_dir / f"query_list_info_seeking.txt", "w")
        for query in query_list:
            fw.write(query + "\n")
        fw.close()
        logger.info("Query list saved to: {}".format(list_root_dir / f"query_list_info_seeking.txt"))
        
    elif command == "plot_histogram":
        data, entity_type_list = [], []
        for split in ["train", "test"]:
            data.extend(read_jsonl(root_dir / f"NERetrive_IR_{split}.jsonl"))
            entity_type_list.extend([line.strip('\n').strip() for line in open(list_root_dir / f"entity_type_list_{split}.txt")])
        logger.info("Start plotting histogram")
        query_list_2 = plot_histogram(data, entity_type_list, num_samples=2)
        logger.info("Finished plotting histogram")
        
        fw = open(list_root_dir / f"query_list_2.txt", "w")
        for query in query_list_2:
            fw.write(query + "\n")
        fw.close()
        
        logger.info("Start plotting histogram for 3 entity types")
        query_list_3 = plot_histogram(data, entity_type_list, num_samples=3)
        logger.info("Finished plotting histogram")
        fw = open(list_root_dir / f"query_list_3.txt", "w")
        for query in query_list_3:
            fw.write(query + "\n")
        fw.close()
        
        

    elif command == "process_count_dict":
        threshold = 100
        num_docs_counter_2 = json.load(open("num_docs_counter_2.json"))
        # print(num_docs_counter_2)
        total_num_docs_2 = sum(num_docs_counter_2.values())
        print(total_num_docs_2)
        num_queries_less_than_threshold_docs = sum([value for key, value in num_docs_counter_2.items() if int(key) < threshold])
        print(num_queries_less_than_threshold_docs)
        logger.info(f"Total number of queries more than {threshold} docs: {total_num_docs_2 - num_queries_less_than_threshold_docs}")
        
        num_docs_counter_3 = json.load(open("num_docs_counter_3.json"))
        total_num_docs_3 = sum(num_docs_counter_3.values())
        print(total_num_docs_3)
        num_queries_less_than_threshold_docs = sum([value for key, value in num_docs_counter_3.items() if int(key) < threshold])
        print(num_queries_less_than_threshold_docs)
        logger.info(f"Total number of queries more than {threshold} docs: {total_num_docs_3 - num_queries_less_than_threshold_docs}")
    elif command == "process_corpus":
        logger.info("Start processing corpus")
        corpus_path = root_dir / f"NERetrive_IR_corpus.jsonl"
        process_corpus(corpus_path)
        logger.info("Finished processing corpus")
