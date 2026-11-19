import json
import argparse
from tqdm import tqdm
    
######### has answer #########
import unicodedata
import string
import regex
def _normalize(text):
    return unicodedata.normalize('NFD', text)

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

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        for i in range(0, len(text) - len(answer) + 1):
            if answer == text[i: i + len(answer)]:
                return True
    return False

def compute_overlap(answer, text, tokenizer):
    answer = _normalize(answer)
    answer = tokenizer.tokenize(answer, uncased=True)
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)
    return len(set(answer) & set(text)) / len(set(answer))


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
###############################


def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def write_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for inst in data:
            f.write(json.dumps(inst) + '\n')

def get_answers(annotations):
    answers = []
    for a in annotations:
        if a['type'] == 'singleAnswer':
            answers.extend(a['answer'])
        else:
            raise ValueError(f'Unknown answer type: {a['type']}')
    return answers

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--command', type=str, default='get_single_answer_data', choices=['get_single_answer_data', 'combine_evidences', 'sort_data_by_score', 'mapping', 'check_evidences', 'filter_data', 'write_to_question_only'])
    args = parser.parse_args()
    
    if args.command == 'get_single_answer_data':
        data_path = 'data/ambiguous/dev_with_evidence_articles.json'
        data = read_json(data_path)
        # print(data)
        print(data[0].keys())
        disagree = 0
        single = 0
        single_data = []
        single_retrieval_data = []
        for inst in data:
            _types = [a['type'] for a in inst['annotations']]
            disagree_inst = False
            if 'singleAnswer' in _types:
                for ans in inst['annotations']:
                    if ans['type'] != 'singleAnswer':
                        print(inst['question'], inst['annotations'])
                        disagree_inst = True
                        
                if not disagree_inst:
                    single += 1
                    single_data.append(inst)
                    single_retrieval_data.append({
                        'question': inst['question'],
                        'ctxs': [],
                        'answers': ['']
                    })
                else:
                    disagree += 1
        print(f'single: {single}')
        print(f'disagree: {disagree}')

        write_jsonl(single_data, 'data/ambiguous/single_qa/ambigqa_dev_single_answer_with_evidence.jsonl')
        write_jsonl(single_retrieval_data, 'data/ambiguous/single_qa/ambigqa_dev_single_answer_question_only.jsonl')
    elif args.command == 'sort_data_by_score':
        for r_idx, retrievers in enumerate(['contriever', 'stella', 'inf', 'nv-embed']):
            data_path = f'results/base_retrievers/{retrievers}/ambigqa_dev_single_answer_question_only.jsonl'
            data = read_jsonl(data_path)
            for inst in data:
                inst['ctxs'] = sorted(inst['ctxs'], key=lambda x: float(x['score']), reverse=True)[:500]
            write_jsonl(data, data_path.replace('.jsonl', f'_sorted.jsonl'))
        
    elif args.command == 'combine_evidences':
        combined_data = []
        # for r_idx, retrievers in enumerate(['contriever', 'stella', 'inf', 'nv-embed', 'bm25']):
        for r_idx, retrievers in enumerate(['contriever', 'stella', 'inf', 'nv-embed']):
            data_path = f'results/base_retrievers/{retrievers}/ambigqa_dev_single_answer_question_only_sorted.jsonl'
            data = read_jsonl(data_path)
            if r_idx == 0:
                combined_data = data
                continue
            print('---')
            print('data', len(data[0]['ctxs']))
            for inst, combined_inst in zip(data, combined_data):
                combined_inst['ctxs'] = combined_inst['ctxs'] + inst['ctxs']
            print('combined_data', len(combined_data[0]['ctxs']))    
            
        write_jsonl(combined_data, 'data/ambiguous/single_qa/ambigqa_dev_single_answer_retrieved_500_combined.jsonl')
        
    elif args.command == 'mapping':
        data_dir = 'data/ambiguous/single_qa'
        retrieved_data_path = f'{data_dir}/ambigqa_dev_single_answer_retrieved_500_combined.jsonl'
        raw_data_path = f'{data_dir}/ambigqa_dev_single_answer_with_evidence.jsonl'
        
        
        
        retrieved_data = read_jsonl(retrieved_data_path)
        raw_data = read_jsonl(raw_data_path)
        assert len(retrieved_data) == len(raw_data)
        # print(raw_data[0]['annotations'][0].keys())
        # print(retrieved_data[0].keys())
        # print(raw_data[0]['annotations'][0])
        
        tokenizer = SimpleTokenizer()
        no_gt_cnt = 0
        for retrieved_inst, raw_inst in tqdm(zip(retrieved_data, raw_data)):
            answers = get_answers(raw_inst['annotations'])
            raw_inst['ground_truths'] = []
            for ctx in retrieved_inst['ctxs']:
                if has_answer(answers, ctx['title'] + ' ' + ctx['text'], tokenizer):
                    # if has_answer([ctx['title'] + ' ' + ctx['text']], '\n'.join(raw_inst['articles_plain_text']), tokenizer):
                    if compute_overlap(ctx['title'] + ' ' + ctx['text'], '\n'.join(raw_inst['articles_plain_text']), tokenizer) > 0.8:
                        ctx['is_gold'] = True
                        raw_inst['ground_truths'].append(ctx)
                else:
                    ctx['is_gold'] = False
            if len(raw_inst['ground_truths']) == 0:
                no_gt_cnt += 1
                
        raw_data = [c for c in raw_data if len(c['ground_truths']) > 0]
        for inst in raw_data:
            inst['ground_truths'] = [inst['ground_truths']]
        print(f'no_gt_cnt: {no_gt_cnt}')
        print(f'total: {len(raw_data)}')
        write_jsonl(retrieved_data, f'{data_dir}/ambigqa_dev_single_answer_retrieved_500_combined.jsonl')
        write_jsonl(raw_data, f'{data_dir}/ambigqa_dev_single_answer_with_gt_filtered.jsonl')

    elif args.command == 'check_evidences':
        data_dir = 'data/ambiguous/single_qa'
        raw_data_path = f'{data_dir}/ambigqa_dev_single_answer_with_evidence.jsonl'
        raw_data = read_jsonl(raw_data_path)
        for inst in raw_data:
            
            print(inst['question'])
            for c in inst['articles_plain_text']:
                print('---')
                print(c)
            print('-' * 100)
            print(len(inst['articles_plain_text']))
            print(inst.keys())
            break
        
    elif args.command == 'filter_data':
        
        def get_single_answer(annotations):
            answers = []
            for a in annotations[:1]:
                if a['type'] == 'singleAnswer':
                    answers.extend(a['answer'])
                else:
                    raise ValueError(f'Unknown answer type: {a['type']}')
            return answers

        data_dir = 'data/ambiguous/single_qa'
        raw_data_path = f'{data_dir}/ambigqa_dev_single_answer_with_gt_filtered.jsonl'
        raw_data = read_jsonl(raw_data_path)
        for inst in raw_data:
            inst['answers'] = [get_answers(inst['annotations'])]
            print(inst['answers'])
        write_jsonl(raw_data, f'{data_dir}/ambigqa_dev_single_answer_with_gt_filtered.jsonl')
        
        print(len(raw_data))
        qs = [inst['question'] for inst in raw_data]
        
        for r_idx, retrievers in enumerate(['contriever', 'stella', 'inf', 'nv-embed']):
            data_path = f'results/base_retrievers/{retrievers}/ambigqa_dev_single_answer_question_only_sorted.jsonl'
            data = read_jsonl(data_path)
            
            data = [inst for inst in data if inst['question'] in qs]
            assert len(data) == len(qs)
            for inst, raw_inst in zip(data, raw_data):
                assert inst['question'] == raw_inst['question']
                inst['answers'] = raw_inst['answers']
                
            write_jsonl(data, f'results/base_retrievers/{retrievers}/ambigqa_dev_single_answer_question_only_sorted_filtered.jsonl')
            
    elif args.command == 'write_to_question_only':
        data_dir = 'data/ambiguous/single_qa'
        raw_data_path = f'{data_dir}/ambigqa_dev_single_answer_with_gt_filtered.jsonl'
        raw_data = read_jsonl(raw_data_path)
        new_data = []
        for inst in raw_data:
            new_data.append({'question': inst['question'], 'input': inst['question'], 'answers': [''], 'ctxs': []})
        write_jsonl(new_data, f'/scratch/hc3337/projects/autoregressive/data/questions/ambigqa_dev_single_answer_question_only.jsonl')
        print(len(new_data))