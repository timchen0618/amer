import json
from ast import literal_eval
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from pathlib import Path
import csv
def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def write_csv(data, file_path):
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)
                        
def main():
    domains = json.load(open('outputs/domains.json', 'r'))
    for domain, sub_domains in tqdm(domains.items()):
        for sub_domain in sub_domains:
            try:
                json.load(open(f'vllm_outputs/questions_woctx/{domain}_{sub_domain}_questions.json', 'r'))['questions']
                # print('Questions already processed for %s %s' % (domain, sub_domain))
            except:
                data = open(f'vllm_outputs/questions_woctx/{domain}_{sub_domain}_questions.json', 'r').read()
                data = data.replace('```json', '').replace('```', '')
                try:
                    data = literal_eval(data)
                    data = json.loads(data)
                    write_json(data, f'vllm_outputs/questions_woctx/{domain}_{sub_domain}_questions.json')
                except:
                    try:
                        data = "?".join(data.split("?")[:-1]) + "?\"]}\""
                        print(data)
                        data = literal_eval(data)
                        data = json.loads(data)
                        write_json(data, f'vllm_outputs/questions_woctx/{domain}_{sub_domain}_questions.json')
                    except:
                        print('Error parsing data for %s %s' % (domain, sub_domain))
                    # print('Error parsing data for %s %s' % (domain, sub_domain))
            

def output_csv():
    domains = json.load(open('outputs/domains.json', 'r'))
    all_data = [['domain', 'sub_domain', 'question']]
    for domain, sub_domains in tqdm(domains.items()):
        for sub_domain in sub_domains:
            data = json.load(open(f'outputs/questions_woctx/{domain}_{sub_domain}_questions.json', 'r'))['questions']
            for inst in data:
                all_data.append([domain, sub_domain, inst])
    
    write_csv(all_data, 'outputs/questions_woctx.csv')
        

def output_docs_csv(output_dir='outputs/q_docs_woctx'):
    domains = json.load(open('outputs/domains.json', 'r'))
    all_data = [['domain', 'sub_domain', 'question', 'documents']]
    
    i = 0
    for domain, sub_domains in tqdm(domains.items()):
        for sub_domain in sub_domains:
            data = json.load(open(f'{output_dir}/{domain}_{sub_domain}_q_and_docs.json', 'r'))
            for inst in data:
                if not isinstance(inst['positive_documents'], list):
                    continue
                all_data.append([domain, sub_domain, inst['question']] + inst['positive_documents'])
            i += 1
            if i > 2:
                break
        
    write_csv(all_data, f'{output_dir}.csv')
        

def process_docs(output_dir='vllm_outputs/q_docs_woctx_1'):
    domains = json.load(open('outputs/domains.json', 'r'))
    print(len(list(Path(output_dir).glob('*.json'))))
    num_non_list = 0
    total_num = 0
    i = 0
    for domain, sub_domains in tqdm(domains.items()):
        for sub_domain in sub_domains:
            data = json.load(open(f'{output_dir}/{domain}_{sub_domain}_q_and_docs.json', 'r'))
            total_num += len(data)
            for inst in data:
                if not isinstance(inst['positive_documents'], list):
                    num_non_list += 1
                    # print(inst['positive_documents'])
        i += 1
    print(f'{output_dir} has {num_non_list} instances with non-list positive documents, out of total {total_num} instances')
            # write_json(data, f'{output_dir}/{domain}_{sub_domain}_q_and_docs.json')


def stats(output_dirs=['outputs/q_docs_wctx_1/', 'outputs/q_docs_woctx_1/', 'outputs/q_docs_wctx/', 'outputs/q_docs_woctx/', 'outputs/q_docs_existing/', 'outputs/q_docs_existing_1/']):
    domains = json.load(open('outputs/domains.json', 'r'))
    for output_dir in output_dirs:
        num_words_list_questions = []
        num_words_list_docs = []
        questions = []
        docs = []
        
        if 'existing' in output_dir:
            data = read_jsonl(f'{output_dir}/existing_q2docs_1k.jsonl')
            # q_data = read_jsonl('../data/eli5+researchy_questions_1k.jsonl')[:200]
            questions += [inst['question'] for inst in data]
            
            for inst in data:
                if len(inst['positive_documents']) > 0 and isinstance(inst['positive_documents'], list):
                    docs += inst['positive_documents']
        else:
            i = 0
            for domain, sub_domains in tqdm(domains.items()):
                for sub_domain in sub_domains:
                    data = json.load(open(f'{output_dir}/{domain}_{sub_domain}_q_and_docs.json', 'r'))
                    questions += [inst['question'] for inst in data]
                    
                    for inst in data:
                        if len(inst['positive_documents']) > 0 and isinstance(inst['positive_documents'], list):
                            docs += inst['positive_documents']
                i += 1
                if i >= 20:
                    break
        num_questions = len(questions)
        num_docs = len(docs)
        num_words_list_questions += [len(word_tokenize(question)) for question in questions]
        num_words_list_docs += [len(word_tokenize(doc)) for doc in docs]
        print(f'{output_dir} has {num_questions} questions and {num_docs} documents')
        print(f'{output_dir} has {num_docs/float(num_questions):.2f} documents per question')
        print(f'{output_dir} has {sum(num_words_list_questions)/float(num_questions):.2f} words per question')
        print(f'{output_dir} has {sum(num_words_list_docs)/float(num_docs):.2f} words per document')

    def get_domains():
        domains = json.load(open('outputs/domains.json', 'r'))
        all_data = [['domain', 'sub_domain']]
        num_domains = 0
        num_sub_domains = 0
        for domain, sub_domains in tqdm(domains.items()):
            all_data.append([domain, ''])
            num_domains += 1
            for sub_domain in sub_domains:
                all_data.append(['', sub_domain])
                num_sub_domains += 1
        print('total number of domains', num_domains)
        print('total number of sub_domains', num_sub_domains)
        write_csv(all_data, 'outputs/domains.csv')
        
if __name__ == '__main__':
    # # main()
    # # output_csv()
    # # output_docs_csv('vllm_outputs/q_docs_woctx_2')
    # # output_docs_csv('vllm_outputs/q_docs_woctx_3')
    stats(output_dirs=['vllm_outputs/q_docs_existing_1/'])
    
    # # process_docs('vllm_outputs/q_docs_woctx_1')
    # process_docs('vllm_outputs/q_docs_woctx_4_100')
    # process_docs('vllm_outputs/q_docs_woctx_4_200')
    # process_docs('vllm_outputs/q_docs_woctx_4_150')
    
    # # stats()
    
    