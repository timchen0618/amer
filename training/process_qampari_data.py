import json

def write_jsonl(data, file_path):
    """
    Write a list of dictionaries to a JSONL file.

    :param data: List of dictionaries to write to the file
    :param file_path: Path to the JSONL file
    """
    with open(file_path, 'w') as file:
        for entry in data:
            json_line = json.dumps(entry)
            file.write(json_line + '\n')

def read_jsonl(file_path):
    """
    Read a JSONL file and return a list of dictionaries.

    :param file_path: Path to the JSONL file
    :return: List of dictionaries
    """
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    
import csv
def read_tsv(file_path):
    with open(file_path, 'r') as file:
        return [line for line in csv.reader(file, delimiter='\t')]
    
def write_tsv(data, file_path):
    with open(file_path, 'w') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(data)
        
def append_tsv(data, file_path):
    with open(file_path, 'a') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(data)
        
        
"""
adapted from chemdataextractor.text.normalize
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tools for normalizing text.
https://github.com/mcs07/ChemDataExtractor
:copyright: Copyright 2016 by Matt Swain.
:license: MIT

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

#: Control characters.
CONTROLS = {
    '\u0001', '\u0002', '\u0003', '\u0004', '\u0005', '\u0006', '\u0007', '\u0008', '\u000e', '\u000f', '\u0011',
    '\u0012', '\u0013', '\u0014', '\u0015', '\u0016', '\u0017', '\u0018', '\u0019', '\u001a', '\u001b',
}
# There are further control characters, but they are instead replaced with a space by unicode normalization
# '\u0009', '\u000a', '\u000b', '\u000c', '\u000d', '\u001c',  '\u001d', '\u001e', '\u001f'


#: Hyphen and dash characters.
HYPHENS = {
    '-',  # \u002d Hyphen-minus
    '‐',  # \u2010 Hyphen
    '‑',  # \u2011 Non-breaking hyphen
    '⁃',  # \u2043 Hyphen bullet
    '‒',  # \u2012 figure dash
    '–',  # \u2013 en dash
    '—',  # \u2014 em dash
    '―',  # \u2015 horizontal bar
}

#: Minus characters.
MINUSES = {
    '-',  # \u002d Hyphen-minus
    '−',  # \u2212 Minus
    '－',  # \uff0d Full-width Hyphen-minus
    '⁻',  # \u207b Superscript minus
}

#: Plus characters.
PLUSES = {
    '+',  # \u002b Plus
    '＋',  # \uff0b Full-width Plus
    '⁺',  # \u207a Superscript plus
}

#: Slash characters.
SLASHES = {
    '/',  # \u002f Solidus
    '⁄',  # \u2044 Fraction slash
    '∕',  # \u2215 Division slash
}

#: Tilde characters.
TILDES = {
    '~',  # \u007e Tilde
    '˜',  # \u02dc Small tilde
    '⁓',  # \u2053 Swung dash
    '∼',  # \u223c Tilde operator #in mbert vocab
    '∽',  # \u223d Reversed tilde
    '∿',  # \u223f Sine wave
    '〜',  # \u301c Wave dash #in mbert vocab
    '～',  # \uff5e Full-width tilde #in mbert vocab
}

#: Apostrophe characters.
APOSTROPHES = {
    "'",  # \u0027
    '’',  # \u2019
    '՚',  # \u055a
    'Ꞌ',  # \ua78b
    'ꞌ',  # \ua78c
    '＇',  # \uff07
}

#: Single quote characters.
SINGLE_QUOTES = {
    "'",  # \u0027
    '‘',  # \u2018
    '’',  # \u2019
    '‚',  # \u201a
    '‛',  # \u201b

}

#: Double quote characters.
DOUBLE_QUOTES = {
    '"',  # \u0022
    '“',  # \u201c
    '”',  # \u201d
    '„',  # \u201e
    '‟',  # \u201f
}

#: Accent characters.
ACCENTS = {
    '`',  # \u0060
    '´',  # \u00b4
}

#: Prime characters.
PRIMES = {
    '′',  # \u2032
    '″',  # \u2033
    '‴',  # \u2034
    '‵',  # \u2035
    '‶',  # \u2036
    '‷',  # \u2037
    '⁗',  # \u2057
}

#: Quote characters, including apostrophes, single quotes, double quotes, accents and primes.
QUOTES = APOSTROPHES | SINGLE_QUOTES | DOUBLE_QUOTES | ACCENTS | PRIMES

def normalize(text):
    for control in CONTROLS:
        text = text.replace(control, '')
    text = text.replace('\u000b', ' ').replace('\u000c', ' ').replace(u'\u0085', ' ')

    for hyphen in HYPHENS | MINUSES:
        text = text.replace(hyphen, '-')
    text = text.replace('\u00ad', '')

    for double_quote in DOUBLE_QUOTES:
        text = text.replace(double_quote, '"')  # \u0022
    for single_quote in (SINGLE_QUOTES | APOSTROPHES | ACCENTS):
        text = text.replace(single_quote, "'")  # \u0027
    text = text.replace('′', "'")     # \u2032 prime
    text = text.replace('‵', "'")     # \u2035 reversed prime
    text = text.replace('″', "''")    # \u2033 double prime
    text = text.replace('‶', "''")    # \u2036 reversed double prime
    text = text.replace('‴', "'''")   # \u2034 triple prime
    text = text.replace('‷', "'''")   # \u2037 reversed triple prime
    text = text.replace('⁗', "''''")  # \u2057 quadruple prime

    text = text.replace('…', '...').replace(' . . . ', ' ... ')  # \u2026

    for slash in SLASHES:
        text = text.replace(slash, '/')

    #for tilde in TILDES:
    #    text = text.replace(tilde, '~')

    return text

import regex
import string


def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(normalize(s)))))


from functools import lru_cache
from operator import itemgetter
import json
import numpy as np
from collections import defaultdict
import argparse
from tqdm import tqdm

def longest_common_substring(x: str, y: str):
    # function to find the longest common substring

    # Memorizing with maximum size of the memory as 1
    @lru_cache(maxsize=1)
    # function to find the longest common prefix
    def longest_common_prefix(i: int, j: int) -> int:

        if 0 <= i < len(x) and 0 <= j < len(y) and x[i] == y[j]:
            return 1 + longest_common_prefix(i + 1, j + 1)
        else:
            return 0

    # diagonally computing the subproblems
    # to decrease memory dependency
    def digonal_computation():

        # upper right triangle of the 2D array
        for k in range(len(x)):
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                        range(len(y) - 1, -1, -1)))

        # lower left triangle of the 2D array
        for k in range(len(y)):
            yield from ((longest_common_prefix(i, j), i, j)
                        for i, j in zip(range(k, -1, -1),
                                        range(len(x) - 1, -1, -1)))

    # returning the maximum of all the subproblems
    return max(digonal_computation(), key=itemgetter(0), default=(0, 0, 0))


from pathlib import Path
import regex
from tqdm import tqdm
import time
from tqdm import tqdm



class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


def get_title2doc(data, title2doc):
    title2doc_all_inst = []
    for inst in tqdm(data):
        title2doc_per_inst = {}
        for ans in inst['answer_list']:
            proof_title = ans['proof'][0]['found_in_url'].split('/')[-1].strip().replace('_', ' ')
            proof_title = normalize_answer(proof_title)
            ans_in_docs = proof_title in title2doc
            
            if not ans_in_docs:
                print(proof_title)
            else:
                title2doc_per_inst[proof_title] = title2doc[proof_title]
                
        title2doc_all_inst.append(title2doc_per_inst)
    return title2doc_all_inst


from pathlib import Path
rootdir = Path('/datastor1/hungting/wikipedia_chunks/chunks_v5')
# Example usage:

file_paths = list(rootdir.glob('*.jsonl'))
print(len(file_paths))
def read_jsonl_sequential(file_paths):
    results = []
    for file_path in tqdm(file_paths):
        results += (read_jsonl(file_path))
    return results


    
tokenizer = SimpleTokenizer()



"""
Process for DPR Corpus
"""

#### use DPR corpus
corpus_name = 'dpr'
print('reading wikipedia corpus')
# get the wikipedia corpus 
corpus = read_tsv('/scratch/cluster/hungting/projects/ODQA/DPR/downloads/data/wikipedia_split/psgs_w100.tsv')

#### Get Title 2 Doc Mapping
print('getting title 2 doc')
title2doc = {}
for row in corpus:
    if normalize_answer(row[2]) not in title2doc:
        title2doc[normalize_answer(row[2])] = []
    title2doc[normalize_answer(row[2])].append({"title": row[2], "text": row[1], "id": row[0]})

#### Get Title 2 Doc for each instance
for split in ['dev', 'train']:
    # read qampari data
    data = read_jsonl(f'/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/{split}_data.jsonl') 
    title2doc_all_inst = get_title2doc(data, title2doc)
    # write to file
    out_data_path = f"/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/{split}_data_gt_{corpus_name}_corpus.jsonl"
    write_jsonl(title2doc_all_inst, out_data_path.replace('.jsonl', '_title2doc.jsonl'))
    
    
"""
Process for QAMPARI Corpus
"""

#### use their corpus
corpus_name = 'qampari'
print('reading QAMPARI wikipedia corpus')
start_time = time.time()
corpus = read_jsonl_sequential(file_paths)
sequential_time = time.time() - start_time
print(f"Sequential reading time: {sequential_time} seconds")

#### Get Title 2 Doc Mapping
title2doc = {}
for row in corpus:
    if normalize_answer(row['meta']['title']) not in title2doc:
        title2doc[normalize_answer(row['meta']['title'])] = []
    title2doc[normalize_answer(row['meta']['title'])].append({"title": row['meta']['title'], "text": row['contents'], "id": row['id']})

#### Get Title 2 Doc for each instance
for split in ['dev', 'train']:
    # read qampari data
    data = read_jsonl(f'/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/{split}_data.jsonl') 
    title2doc_all_inst = get_title2doc(data, title2doc)
    # write to file
    out_data_path = f"/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/{split}_data_gt_{corpus_name}_corpus.jsonl"
    write_jsonl(title2doc_all_inst, out_data_path.replace('.jsonl', '_title2doc.jsonl'))
    



split = 'train'
corpus_name = 'qampari'
MAX_SCORE_THRESHOLD = 0.3
NUM_ANSWERS=4

data = read_jsonl(f'/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/{split}_data.jsonl') 
title2docs = read_jsonl(f'/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/{split}_data_gt_{corpus_name}_corpus_title2doc.jsonl')
assert len(data) == len(title2docs), (len(data), len(title2docs))

proof_link_in_corpus = 0
new_data = []
csv_data = [['score', 'proof_text', 'candidate_doc']]
less_than_4 = 0

for inst in tqdm(data):
    ground_truths = []
    title2doc_per_inst = title2docs.pop(0)  # for each instance, get the title 2 docs mapping
    if len(inst['answer_list']) < NUM_ANSWERS:
        less_than_4 += 1
        continue
    for ans in inst['answer_list']:
        # collect proof title
        proof_title = ans['proof'][0]['found_in_url'].split('/')[-1].strip().replace('_', ' ')
        proof_title = normalize_answer(proof_title)
        # check if this answer can be mapped
        ans_in_docs = proof_title in title2doc_per_inst
        
        if not ans_in_docs:
            continue
        else:
            candidate_docs = title2doc_per_inst[proof_title]
            scores = []
            for doc in candidate_docs:  # compute score for each candidate doc
                length, _, _ = longest_common_substring(normalize_answer(ans['proof'][0]['proof_text'].lower()).replace('\n', ''), normalize_answer(doc['text'].lower()).replace('\n', ''))
                scores.append(length / len(normalize_answer(ans['proof'][0]['proof_text'])))
            
            best_doc_idx = scores.index(max(scores)) # get the best index
            
            if max(scores) < MAX_SCORE_THRESHOLD:  # if the best score is below threshold, add to csv, else add to ground truths
                csv_data.append([max(scores), ans['proof'][0]['proof_text'], candidate_docs[best_doc_idx]['text'], normalize_answer(ans['proof'][0]['proof_text'].lower()), normalize_answer(candidate_docs[best_doc_idx]['text'].lower())])
            else:
                ground_truths.append(candidate_docs[best_doc_idx])
                
            
    if len(ground_truths) >= NUM_ANSWERS:  # if more than 4 answers are in the corpus, add to new data
        proof_link_in_corpus += 1
        inst['ground_truths'] = ground_truths
        new_data.append(inst)

write_tsv(csv_data, 'out.tsv') 
print(f'All proof links are in the corpus for {proof_link_in_corpus} questions')
print('less than 4 answers:', less_than_4)
print(proof_link_in_corpus / len(data))

out_data_path = f"/scratch/cluster/hungting/projects/diverse_response/data/qampari_data/{split}_data_gt_{corpus_name}_corpus.jsonl"
write_jsonl(new_data, out_data_path)
