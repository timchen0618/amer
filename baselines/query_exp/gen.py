from openai import OpenAI
import json
from pathlib import Path
import csv
from tqdm import tqdm

# def pred_gpt4(client, prompt):
#     messages = [{"role": "user", "content": prompt}]
        
#     response = client.chat.completions.create(
#     model="gpt-4o-2024-08-06",
#     messages=messages
#     )
#     response = response.choices[0].message.content.strip()
#     return response


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

rootdir = '/scratch/hc3337/projects/autoregressive/results/base_retrievers/inf/'
data_types = ['ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs', 'dev_data_gt_qampari_corpus_5_to_8_ctxs']

for data_type in data_types:
    input_file = f'{rootdir}/{data_type}.json'

    out_data = []
    out_file = Path(input_file).stem + '_query_exp.jsonl'

    data = read_jsonl(input_file)
    if data_type == 'ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs':
        data = data[34:]
    elif data_type == 'dev_data_gt_qampari_corpus_5_to_8_ctxs':
        data = data[3:]
        
    client = OpenAI()

    for inst in tqdm(data):
        # try:
        response = pred_gpt4(client, inst['question'])
        # except:
        #     print('skipped question', inst['question'])
        #     continue
        out_data.append(inst)
        out_data[-1]['question'] = f'Question: {inst["question"]} \nRelevant Keywords: {response}'
        
        print('writing to', out_file)
        write_jsonl_line(out_file, inst)

# write_jsonl(out_file, data)