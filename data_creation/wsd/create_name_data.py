import json

def read_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def write_jsonl(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')

names = open('names.txt').read().strip('\n').strip().split(',')
responses = read_jsonl('responses.jsonl')
assert len(names) == len(responses)
print(len(names), len(responses))
corpus = []
_id = 0

out = []
out_alphabet = []
qid = 0
alphabet2docs = {}
# {"question", "ground_truths", "id"}
for name, response in zip(names, responses):
    if isinstance(response['documents'][0], str):
        all_doc_string = '\n'.join(response['documents'])
    elif isinstance(response['documents'][0], dict):
        # print(response['documents'][0])
        if 'description' in response['documents'][0]:
            all_doc_string = '\n'.join([d['description'] for d in response['documents']])
        elif 'content' in response['documents'][0]:
            all_doc_string = '\n'.join([d['content'] for d in response['documents']])
    tokenized_string = all_doc_string.split()
    paragraphs = [' '.join(tokenized_string[i:i+100]) for i in range(0, len(tokenized_string), 100)]
    
    question = "Retrieve documents that contain the name " + name + "."
    if name[0] not in alphabet2docs:
        alphabet2docs[name[0]] = []
    positive_docs = []
    # add paragraph to corpus
    for doc in paragraphs:
        doc_json = {
            "id": str(_id),
            "text": doc,
            "title": "",
            }
        corpus.append(doc_json)
        _id += 1    
        
        # identify the positive docs
        if name in doc:
            positive_docs.append([doc_json])
            alphabet2docs[name[0]].append([doc_json])
    out.append({
        "question": question,
        "qid": qid,
        "positive_ctxs": positive_docs,
        "ground_truths": positive_docs,
        "answers": [name],
    })


print(alphabet2docs.keys())
import string
for character in string.ascii_uppercase:
    question = "Retrieve documents that contain names starting with character " + "\"" + character + "\"" + "."
    positive_docs = alphabet2docs[character]
    out_alphabet.append({
        "question": question,
        "qid": qid,
        "positive_ctxs": positive_docs,
        "ground_truths": positive_docs,
        "answers": [character],
    })
    qid += 1
    
write_jsonl('dev.jsonl', out)
write_jsonl('dev_alphabet.jsonl', out_alphabet)


fw = open('corpus.tsv', 'w')
fw.write('id\ttext\ttitle\n')
for inst in corpus:
    fw.write(f"{inst['id']}\t{inst['text']}\t{inst['title']}\n")
fw.close()
    
print(len(corpus))



for inst in out:
    inst['input'] = inst['question']
    inst['id'] = inst['qid']
    inst['ctxs'] = []

fw = open('dev.json', 'w')
json.dump(out, fw, indent=4)
fw.close()


for inst in out_alphabet:
    inst['input'] = inst['question']
    inst['id'] = inst['qid']
    inst['ctxs'] = []

fw = open('dev_alphabet.json', 'w')
json.dump(out_alphabet, fw, indent=4)
fw.close()

avg_lens = []
data = read_jsonl('dev_alphabet.jsonl')
for inst in data:
    avg_lens.append(len(inst['ground_truths']))
print(sum(avg_lens)/len(avg_lens))



