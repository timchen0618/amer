import json
from unicodedata import category

def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def write_tsv(data, file_path, category_list):
    import csv
    with open(file_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['question'] + category_list)
        for item in data:
            documents = [item['selected_golden_documents'][category].get('text', '') if item['selected_golden_documents'][category] else '' for category in category_list]
            writer.writerow([item['query_text']] + documents)

category_list = ["Wikipedia", "Scholarly Articles", "Web (Common Crawl, C4)", "Github", "Math", "Pubmed", "Book", "Stackexchange"]
data = read_json('golden_retrieval_dataset_per_category.json')
write_tsv(data, 'golden_retrieval_dataset_per_category.tsv', category_list)











              