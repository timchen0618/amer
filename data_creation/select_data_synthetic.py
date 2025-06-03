# convert all_questions.txt to nq format
import json
def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data

def write_json(data, file_path):
    """
    Write a dictionary to a JSON file, indent=4.

    :param data: Dictionary to write to the file
    :param file_path: Path to the JSON file
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


# gather all the data into a single "question.txt"
from pathlib import Path

suffix = '_q_sm'
rootdir = Path('/scratch/hc3337/data')
questions = []

# hotpot_data = load_data(str(rootdir / 'hotpot_train_v1.1.json'))
# questions += [inst['question'] for inst in hotpot_data]

# nq_data = load_data(str(rootdir / 'nq-open/NQ-open.train.jsonl'))
# questions += [inst['question'] for inst in nq_data]

qampari_data = load_data('/scratch/hc3337/projects/diverse_response/data/qampari_data/train_data.jsonl')
questions += [inst['question_text'] for inst in qampari_data][:3000]

f = open(f'all_train_questions{suffix}.txt', 'w')
for l in questions:
    f.write(l + '\n')
f.close()



all_questions = [l.strip('\n') for l in open(f'all_train_questions{suffix}.txt')]

data = []
i = 0
for q in all_questions:
    inst = {
        "question": q,
        "input": q,
        'id': str(i),
        'answers': [''],
        'ctxs': []
    }
    data.append(inst)
    i += 1
write_json(data, f'all_train_questions{suffix}.json')

data = json.load(open(f'all_train_questions{suffix}.json'))
print(len(data))