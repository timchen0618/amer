from openai import OpenAI
import json
from pathlib import Path
import csv
from tqdm import tqdm


def pred_gpt4(client, question):
    response = client.responses.create(
    prompt={
        "id": "pmpt_68acd37437a88196a02e00daad4c57f50d7ec9cab5747dd3",
        "version": "1",
        "variables": {
        "question": question
        }
    }
    )
    return response.output_text

def read_jsonl(filename):
    data = []
    with open(filename, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))
    return data

def write_jsonl(filename, data):
    with open(filename, 'w') as fout:
        for d in data:
            fout.write(json.dumps(d) + '\n')

def write_jsonl_line(filename, data):
    with open(filename, 'a') as fout:
        fout.write(json.dumps(data) + '\n')

def form_prompt(instruction, input_text):
    return instruction.replace('[Question]', input_text)

def form_prompt_winput(instruction, question, documents):
    return instruction.replace('[Question]', question).replace('[Documents]', documents)

def write_tsv(filename, data):
    with open(filename, 'w', newline='') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerows(data)


instruction = open('instruction_keywords.txt', 'r').read()

project_dir = '/path/to/project'

# base
# rootdir = f'{project_dir}/results/base_retrievers/inf/'
# data_types = ['ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs', 'dev_data_gt_qampari_corpus_5_to_8_ctxs']

# # stage 1 qampari
# rootdir = f'{project_dir}/results/llama-1b/qampari_inf/toy_qemb_from_nq/'
# data_types = ['retrieval_out_dev_qampari_single.jsonl']


# stage 2 nq
rootdir = f'{project_dir}/results/llama-1b/nq_inf/toy_contrastive/'
data_types = ['retrieval_out_dev_ambiguous_qe_single.jsonl']

for data_type in data_types:
    input_file = f'{rootdir}/{data_type}'

    out_data = []
    out_file = Path(input_file).stem + '_query_exp.jsonl'

    data = read_jsonl(input_file)
        
    client = OpenAI()

    for inst in tqdm(data):
        question = inst['question'] if 'question' in inst else inst['question_text']
        response = pred_gpt4(client, question)
        out_data.append(inst)
        out_data[-1]['question'] = f'Question: {question} \nRelevant Keywords: {response}'
        
        print('writing to', out_file)
        write_jsonl_line(out_file, inst)

