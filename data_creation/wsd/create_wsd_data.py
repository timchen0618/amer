from dis import Instruction
from re import L
from nltk.corpus import wordnet as wn
from openai import OpenAI
from tqdm import tqdm
import json
from pathlib import Path

client = OpenAI()

def write_jsonl(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')
            
def write_one_line(filename, line):
    with open(filename, 'a') as f:
        f.write(json.dumps(line) + '\n')

def read_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

            
def get_completion(client, prompt):
  completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": prompt}
      ],
      response_format={ "type": "json_object" },
  )
  return completion.choices[0].message.content

import csv
def read_tsv(filepath='data/corpus.tsv', delimiter='\t'):
    corpus = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader)  # Skip header
        for row in reader:
            doc = {
                'id': row[0],
                'text': row[1],
                'title': row[2]
            }
            corpus.append(doc)
    return corpus


def save_word2sense(collect_all_sense=False):
    all_data_path = '../data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL.gold.key.txt'
    all_data = open(all_data_path).readlines()

    if collect_all_sense:
        # collect all words in the benchmark
        words = set()
        for inst in tqdm(all_data):
            for key in inst.split()[1:]:
                word = key.split('%')[0]
                words.add(word)
                
        word2sense = {}
        for word in words:
            word2sense[word] = [syn.definition() for syn in wn.synsets(word)]
        
    else:        
        word2sense = {}
        for inst in tqdm(all_data):
            keys = inst.split()[1:]
            for key in keys:
                word = key.split('%')[0]
                if word not in word2sense:
                    word2sense[word] = set()
                # Convert sense key to WordNet synset
                sense_key = wn.lemma_from_key(key)
                synset = wn.lemma_from_key(key).synset()
                word2sense[word].add(synset.definition())
            
    print(len(word2sense))
    avg_lens = []
    num_words = 0
    keys_to_delete = []
    for k, v in word2sense.items():
        if len(v) == 1:
            keys_to_delete.append(k)
            continue
        avg_lens.append(len(v))
        num_words += 1
        word2sense[k] = list(v)
    
    for k in keys_to_delete:
        del word2sense[k]
    
    print('num_words', num_words, 'avg_len', sum(avg_lens)/float(len(avg_lens)))
    if collect_all_sense:
        fw = open('../data/wsd/words2allsense.json', 'w')
    else:
        fw = open('../data/wsd/words2sense.json', 'w')
    json.dump(word2sense, fw, indent=4)
    fw.close()
    
def save_word2sense_distinct(words_meaning_path):
    words_and_meanings = [l.strip('\n').strip().split(':') for l in open(words_meaning_path, 'r')]
    words_and_meanings = [[l[0], [m.strip() for m in l[1].split('/')]] for l in words_and_meanings]
    word2sense = {}
    for w_and_m in tqdm(words_and_meanings):
        w, m = w_and_m[0], w_and_m[1]
        word2sense[w] = m
    
    print(len(word2sense))
    fw = open('../data/wsd/distinct/words2sense_distinct.json', 'w')
    json.dump(word2sense, fw, indent=4)
    fw.close()
    
def save_word2sense_distinct_avoid_test(words_meaning_path, test_words_path):
    words_and_meanings = [l.strip('\n').strip().split(':') for l in open(words_meaning_path, 'r')]
    words_and_meanings = [[l[0], [m.strip() for m in l[1].split('/')]] for l in words_and_meanings]
    test_words_and_meanings = [l.strip('\n').strip().split(':') for l in open(test_words_path, 'r')]
    test_words_and_meanings = [[l[0], [m.strip() for m in l[1].split('/')]] for l in test_words_and_meanings]
    test_words = set([w for w, _ in test_words_and_meanings])
    
    word2sense = {}
    for w_and_m in tqdm(words_and_meanings):
        w, m = w_and_m[0], w_and_m[1]
        if w not in test_words and w not in word2sense:
            word2sense[w] = m
    
    print(len(word2sense))
    fw = open('../data/wsd/distinct/words2sense_distinct_train.json', 'w')
    json.dump(word2sense, fw, indent=4)
    fw.close()

def get_max_num_senses(word2sense):
    return max([len(v) for k, v in word2sense.items()])


def generate_documents(rootdir=Path('../data/wsd/'), word_sense_dict='words2sense.json', instruction_path='instructions_wsd.txt', out_path='responses_wsd.jsonl', use_number=False):
    assert use_number == ('number' in instruction_path)  # whenever you use a number to define number of documents, you should use the corresponding instruction
    word2sense = json.load(open(rootdir / word_sense_dict))
    max_num_senses = get_max_num_senses(word2sense)
    len_documents = [2 for _ in range(max_num_senses)]
    len_documents[:3] = [10, 5, 3]
    
    instruction = open(instruction_path).read()
    # all_responses = []
    for word, senses in tqdm(word2sense.items()):
        for sense_id, sense in enumerate(senses):
            num_docs = len_documents[sense_id]
            prompt = instruction.replace('[WORD]', word).replace('[DEFINITION]', sense).replace('[NUMBER]', str(num_docs))
            response = get_completion(client, prompt)
            try:
                out_dict = json.loads(response)
                out_dict['word'] = word
                out_dict['sense'] = sense
                write_one_line(out_path, out_dict)
                # all_responses.append(out_dict)
                
            except:
                print('skip...')
                
    # write_jsonl(out_path, all_responses)



def process_documents(response_path='responses_wsd.jsonl', out_dev_path='', out_train_path='', corpus_path='', dev_split_size=0):
    def write_json(filename, data):
        fw = open(filename, 'w')
        json.dump(data, fw, indent=4)
        fw.close()
        
    # keys => dict_keys(['documents', 'word', 'sense'])
    responses = read_jsonl(response_path)
    corpus = []
    _id = 0
    out = []
    
    word2documents = {}
    
    for resp in responses:
        word = resp['word']
        sense = resp['sense']
        documents = resp['documents']
        
        if word not in word2documents:
            word2documents[word] = {}
        if sense not in word2documents[word]:
            word2documents[word][sense] = []
        
        for doc in documents:
            doc_json = {
                "id": str(_id),
                "text": doc,
                "title": "",
                }
            corpus.append(doc_json)
            word2documents[word][sense].append(doc_json)
            _id += 1    
    
    # write corpus 
    fw = open(corpus_path, 'w')
    fw.write('id\ttext\ttitle\n')
    for inst in corpus:
        fw.write(f"{inst['id']}\t{inst['text']}\t{inst['title']}\n")
    fw.close()    
    print('len corpus', len(corpus))
        

    # Create query - positive documents relation
    qid = 0
    num_documents_per_sense = []
    num_senses_per_word = []
    for word, documents_per_sense in word2documents.items():
        question = f"Retrieve documents that describe the word {word}, with various possible definitions."
        positive_documents = []
        train_positive_documents = []
        num_senses_per_word.append(len(documents_per_sense))
        for sense, documents in documents_per_sense.items():
            documents = [d for d in documents if (isinstance(d['text'], str) and d['text'] != '')]
            num_documents_per_sense.append(len(documents))
            positive_documents.append(documents)
            train_positive_documents.append(documents[0])

        out.append({
            "question": question,
            "qid": str(qid),
            "positive_ctxs": positive_documents,
            "ground_truths": positive_documents,
            "answers": [word],
            "senses": [sense for sense, _ in documents_per_sense.items()],
        })
        qid += 1
    print('avg num documents per sense', sum(num_documents_per_sense)/len(num_documents_per_sense))
    print('avg num senses per word', sum(num_senses_per_word)/len(num_senses_per_word))

    if dev_split_size == 0:
        out_dev = out
        out_train = out
    else:
        assert dev_split_size < 1
        import random
        random.shuffle(out)
        out_dev, out_train = out[:int(len(out)*dev_split_size)], out[int(len(out)*dev_split_size):]
    
    write_jsonl(out_dev_path, out_dev)
    write_json(out_dev_path[:-1], out_dev)
    
    for inst in out_train:
        inst['positive_ctxs'] = [c[0] for c in inst['positive_ctxs']]
        inst['ground_truths'] = [c[0] for c in inst['ground_truths']]
    write_jsonl(out_train_path, out_train)
    write_json(out_train_path[:-1], out_train)
    


    
    

def combine_corpus():
    random_negs = read_tsv('../data/wsd/distinct/corpus_train.tsv')
    corpus = read_tsv('../data/wsd/distinct/corpus_small.tsv')
    
    fw = open('../data/wsd/distinct/corpus.tsv', 'w')
    fw.write('id\ttext\ttitle\n')
    for inst in (random_negs + corpus):
        fw.write(f"{inst['id']}\t{inst['text']}\t{inst['title']}\n")
    fw.close()
    
    
def most_frequent():
    from nltk.corpus import wordnet as wn

    word = "ice"

    # Get most frequent sense (first synset)
    most_frequent_synset = wn.synsets(word)[0]
    most_frequent_synset_1 = wn.synsets(word)[1]
    print(wn.synsets(word))
    print()

    print(f"Most Frequent Sense: {most_frequent_synset}")
    print(f"Definition: {most_frequent_synset.definition()}")

    print(f"Most Frequent Sense: {most_frequent_synset_1}")
    print(f"Definition: {most_frequent_synset_1.definition()}")
    

def random_baseline(corpus_path, data_path='../data/wsd/dev.jsonl'):
    corpus = read_tsv(corpus_path)
    corpus_string = Path(corpus_path).stem
    questions = read_jsonl(data_path)
    
    import random
    random.seed(0)
    
    out = []
    for inst in tqdm(questions):
        out.append({"question": inst['question'], "ctxs": random.sample(corpus, 100), "id": inst["qid"]})
        
    write_jsonl(f'../data/wsd/random_baseline_{corpus_string}.jsonl', out)
    
    
def export_to_tsv(input_file, outfile_path):
    out = [['question', 'qid', 'word', 'sense 1', 'positive_ctxs_1', 'sense 2', 'positive_ctxs_2', 'sense 3', 'positive_ctxs_3', 'sense 4', 'positive_ctxs_4', 'sense 5', 'positive_ctxs_5' ]]
    data = read_jsonl(input_file)
    for inst in data:
        out_inst = []
        out_inst.append(inst['question'])
        out_inst.append(inst['qid'])
        out_inst.append(inst['answers'][0])
        for i, ctx in enumerate(inst['positive_ctxs']):
            out_inst.append(inst['senses'][i])
            out_inst.append(json.dumps(ctx))
        out.append(out_inst)
    
    import csv
    with open(outfile_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerows(out)
    





if __name__ == '__main__':
    # save_word2sense(collect_all_sense=True)
    # save_word2sense_distinct('distinct_word_meanings.txt')
    # save_word2sense_distinct_avoid_test('distinct_word_meanings_train.txt', 'distinct_word_meanings.txt')
    
    # generate_documents(rootdir=Path('../data/wsd/distinct'), 
    #                    word_sense_dict='words2sense_distinct_train.json', 
    #                    instruction_path='instructions_wsd_number.txt', 
    #                    out_path='responses_wsd_distinct_train.jsonl',
    #                    use_number=True)
    
    # process_documents(response_path='responses_wsd_all_sense.jsonl', 
    #                   out_dev_path='../data/wsd/all_sense/dev_all_sense.jsonl', 
    #                   out_train_path='../data/wsd/all_sense/train_all_sense.jsonl',
    #                   corpus_path='../data/wsd/all_sense/corpus_all_sense.tsv',
    #                   dev_split_size=0.2)
    # process_documents(response_path='responses_wsd_distinct_train.jsonl', 
    #                   out_dev_path='../data/wsd/distinct/dev_train.jsonl', 
    #                   out_train_path='../data/wsd/distinct/train_train.jsonl',
    #                   corpus_path='../data/wsd/distinct/corpus_train.tsv',
    #                   dev_split_size=0.0)
    
    
    
    combine_corpus()
    # most_frequent()
    # random_baseline('../data/wsd/distinct/corpus.tsv', '../data/wsd/distinct/dev.jsonl')
    
    # export_to_tsv(input_file="../data/wsd/distinct/dev_train.jsonl",
    #               outfile_path="out_dev_data.tsv",
    #               )