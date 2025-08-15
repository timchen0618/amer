# from prediction_mistral import Prediction
import random
from pathlib import Path
import logging
from tqdm import tqdm
import argparse
import ast

import torch
import transformers
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json

def load_model(args, device, logger):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
    from peft import AutoPeftModelForCausalLM, PeftModel, PeftConfig
    from transformers import pipeline
    
    if args.model == 'google/t5_xxl_true_nli_mixture':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info('finish loading model')
        model.to(device)
        return model, tokenizer
    elif args.model.find('Mistral') != -1:
        if args.model == 'timchen0618/Mistral_BERDS_evaluator_full':
            model = AutoModelForCausalLM.from_pretrained(args.model)
        elif args.model.find('saved') != -1 or args.model.find('timchen0618') != -1:
            model = AutoPeftModelForCausalLM.from_pretrained(args.model)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        logger.info('finish loading model')
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
        return model, tokenizer
    elif args.model.find('zephyr') != -1:
        pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device_map="auto")
        logger.info('finish loading model')
        return pipe, None
    
    elif args.model.find('gemma') != -1:        
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto")
        logger.info('finish loading model')
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
        return model, tokenizer
    elif args.model.find('llama') != -1:
        from llama import Llama, Dialog
        import yaml
        with open(args.llama_config, "r") as stream:
            try:
                configs = yaml.safe_load(stream)[args.model]
            except yaml.YAMLError as exc:
                logger.error(exc)
                
        model = Llama.build(
            ckpt_dir=configs['ckpt_dir'],
            tokenizer_path=configs['tokenizer_path'],
            max_seq_len=configs['max_seq_len'],
            max_batch_size=configs['max_batch_size']
        )
        configs = configs['generation']
        logger.info('finish loading model')
        return model, configs
    else:
        raise NotImplementedError
    


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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Inferencer:
    def __init__(self, model_id, n_gpu, instruction):
        self.model_id = model_id
        self.llm = LLM(model=model_id,
                    tokenizer=self.model_id,
                    enable_prefix_caching=True,
                    trust_remote_code=True,
                    tensor_parallel_size=n_gpu,
                    gpu_memory_utilization=0.9)
        self.sampling_params = SamplingParams(temperature=0.0, max_tokens=16, seed=0)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.instruction = instruction
    
    def formulate_dialogs(self, doc_1, doc_2):
        start_tok = '<|im_start|>'
        end_tok = '<|im_end|>'
        _instruction = self.instruction.replace('[Doc 1]', doc_1).replace('[Doc 2]', doc_2)
        messages = _instruction.split(end_tok)[:-1]
        messages = [{"role": m.strip('\n').split('\n')[0][12:], "content": '\n'.join(m.strip('\n').split('\n')[1:]).strip()} for m in messages]    
        return messages
    
    def prepare_input(self, data, topk):
        input_texts = []  # num_instance * num_perspectives * num_docs
        for inst in tqdm(data):
            # retrieving perspectives
            perspectives = inst['perspectives']
            
            # retrieving docs
            docs = inst['ctxs']
                
            # for every perspective, check if it is supported by the docs
            for p in perspectives:
                # pred_inst.append([])
                for doc in docs[:topk]:
                    if 'title' in doc:
                        doc_text = doc['text'] + ' ' + doc['title']
                    else:
                        doc_text = doc['text']   
                    
                    input_texts.append(self.formulate_dialogs(doc_text, p)[1:])
                    
        return input_texts
                        
        
    def inference(self, input_texts):
        self.llm.generate(input_texts[0], self.sampling_params)
    
        outputs = self.llm.generate(input_texts, self.sampling_params, use_tqdm=True)
        outputs = [output.outputs[0].text for output in outputs]
        return outputs
    
    def parse_response(self, response):
        if response[0] != '{':
            response = response.strip().strip('\n').split('Answer:')[-1].split('The answer is')[-1].split('The answer is \"')[-1].split('The answer is:')[-1].split('Based on the information provided in the document, the answer is')[-1].split('Based on the information provided in the document, the answer is:\n')[-1].strip()
            response = response.strip().lower()
            _yes = (response[:3] == 'yes' or response[:4] == '(yes') or (response == 'y' or response == 'ye') or (response[-3:] == 'yes' or response[-4:] == 'yes.' or response[-5:] == 'yes\".')
            _entail_or_true = (response[:6] == 'entail' or (response[:4] == 'true'))
            return  _yes or _entail_or_true
        else:
            return ast.literal_eval(response)['Answer'].strip().lower() == 'yes'

def score_recall(preds):
    # average 
    recalls = []
    for inst in preds:
        recalls.append(sum([any(preds_per_perspective) for preds_per_perspective in inst])/len(inst))

    return sum(recalls) / float(len(recalls))


def score_mrecall(preds):
    mrecalls = []
    topk = len(preds[0][0])
    for inst in preds:
        if len(inst) > topk:
            mrecalls.append(int(sum([any(preds_per_perspective) for preds_per_perspective in inst])>=topk))
        else:
            mrecalls.append(int(all([any(preds_per_perspective) for preds_per_perspective in inst])))
        
    return sum(mrecalls) / float(len(mrecalls))
          

def score_precision(preds):
    precisions = []
    
    topk = len(preds[0][0])
    for inst in preds:
        assert len(inst[0]) >= topk
        num_perspective_containing_docs = 0
        for j in range(topk):
            contain_any_perspective = False
            for p in inst:
                if p[j]:
                    contain_any_perspective = True
                    break
            if contain_any_perspective:
                num_perspective_containing_docs += 1
            
        precisions.append(num_perspective_containing_docs / topk)
        
    return sum(precisions) / float(len(precisions))

def prepare_predictor_and_inputs(args, data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # read instructions
    instruction = open(args.instructions).read()
    inferencer = Inferencer(args.model, 1, instruction)
    
    input_texts = inferencer.prepare_input(data, args.topk)
    input_texts = inferencer.tokenizer.apply_chat_template(input_texts, add_generation_prompt=True, tokenize=False)
    return inferencer, input_texts
    

def main(args):
    random.seed(0)
    torch.manual_seed(0)
    transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
    
    preds = []
    if not args.compute_only:
        """
            Running perspective detection module, and store the results in args.output_file
            
            Data Format (each instance):
            {
                "question": ...,
                "ctxs": [
                    {"title": ..., "text": ...},
                    {"title": ..., "text": ...},
                    ...
                ],
                "perspectives": [
                    "perspective 1",
                    "perspective 2",
                    ...
                ]
            }
            
            After running the perspective detection module, the "ctxs" field of the output file will have the following format:
            "ctxs": [
                {"title": ..., "text": ..., "mistral-preds": [True, False, ...]},
                {"title": ..., "text": ..., "mistral-preds": [True, False, ...]},
                ...
            ],
        """
        
        logger.info('running perspective detection: eval')
        assert Path(args.data).suffix == '.jsonl'
        
        # prepare data
        data = read_jsonl(args.data)
        
        if args.data.find('arguana_generated') != -1:
            data = data[:750]
        elif args.data.find('kialo') != -1:
            data = data[:774]
        elif args.data.find('opinionqa') != -1:
            data = data[:882]
        
        # prepare predictor 
        logger.info('loading model and data...')
        inferencer, input_texts = prepare_predictor_and_inputs(args, data)
        logger.info('finish loading model, start inference...')
        outputs = inferencer.inference(input_texts)
        # print(outputs)
        outputs = [inferencer.parse_response(output) for output in outputs]
        # print(outputs)
        
        
        for inst in tqdm(data):
            pred_inst = []  # (num_perspectives, num_docs)
            # retrieving perspectives
            perspectives = inst['perspectives']
            
            # retrieving docs
            docs = inst['ctxs']
                
            # for every perspective, check if it is supported by the docs
            for p in perspectives:
                pred_inst.append([])
                for doc in docs[:args.topk]:
                    if 'title' in doc:
                        doc_text = doc['text'] + ' ' + doc['title']
                    else:
                        doc_text = doc['text']   
                        
                    pred = outputs.pop(0)
                    if args.model_type == 'gpt4':
                        if 'gpt4-preds' not in doc:
                            doc['gpt4-preds'] = []
                            
                        if len(doc['gpt4-preds']) <= len(perspectives):
                            doc['gpt4-preds'].append(pred)
                    elif args.model_type == 'mistral':
                        if 'mistral-preds' not in doc:
                            doc['mistral-preds'] = []
                            
                        if len(doc['mistral-preds']) <= len(perspectives):
                            doc['mistral-preds'].append(pred)
                    else:
                        raise NotImplementedError

                    pred_inst[-1].append(pred)
            preds.append(pred_inst)    
        
        write_jsonl(args.output_file, data)

    else:
        """
            Not running perspective detection module, only computing scores based on previously perspective detection results
        """
        assert (Path(args.data).name.find('.gpt4pred') != -1 or Path(args.data).name.find('.mistralpred') != -1)
        logger.info("only computing scores based on previously perspective detection results")
        data = read_jsonl(args.data)                   
        
        for inst in tqdm(data):
            pred_inst = []  # (num_perspectives, num_docs)
            perspectives = inst['perspectives']
            docs = inst['ctxs']

            pred_str = 'gpt4-preds' if Path(args.data).name.find('.gpt4pred') != -1 else 'mistral-preds'
            for j, p in enumerate(perspectives):
                pred_inst.append([])
                for doc in docs[:args.topk]:
                    pred = doc[pred_str][j]
                    pred_inst[-1].append(pred)

            preds.append(pred_inst)
            
    # compute Precision & MRecall
    precision_score = score_precision(preds)
    mrecall_score = score_mrecall(preds)
    recall_score = score_recall(preds)
    
    logger.info(f'average precision: {100*precision_score:.2f}')
    logger.info(f'mrecall: {100*mrecall_score:.2f}')
    logger.info(f'recall: {100*recall_score:.2f}')
    
    # writing score to a csv file
    fw = open('score.csv', 'w')
    fw.write(f'\n{100*precision_score:.2f}\n{100*mrecall_score:.2f}\n{100*recall_score:.2f}')
    fw.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/t5_xxl_true_nli_mixture", type=str)
    parser.add_argument("--model_type", default="t5", type=str)
    parser.add_argument("--output_file", type=str, default="output.jsonl.mistralpred")
    parser.add_argument("--instructions", type=str, default="instructions_nli.txt")
    parser.add_argument("--topk", type=int, default=10)  
    parser.add_argument("--compute_only", action='store_true') 
    
    parser.add_argument("--data", type=str, default="/path/to/some/retrieval/output.jsonl")
    args = parser.parse_args()

    main(args)
    
