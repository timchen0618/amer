import json
import pickle
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import torch
from tqdm import tqdm
from pathlib import Path
from vllm import LLM, SamplingParams
from ast import literal_eval
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def write_json(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def write_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def append_jsonl(data, file_path):
    with open(file_path, 'a') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


@torch.no_grad()
def generate(prompts, tokenizer, model, thinking=True, use_vllm=False):
    if use_vllm:
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32768)
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=True)
        contents = [output.outputs[0].text for output in outputs]
        thinking_contents = [None for _ in prompts]
    else:
        contents = []
        thinking_contents = []
        for prompt in prompts:
            model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            # conduct text completion
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=32768
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
            if thinking:
                # parsing thinking contents
                try:
                    # rindex finding 151668 (</think>)
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0

                thinking_content_per_prompt = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content_per_prompt = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            else:
                thinking_content_per_prompt = None
                content_per_prompt = tokenizer.decode(output_ids, skip_special_tokens=True)
            thinking_contents.append(thinking_content_per_prompt)
            contents.append(content_per_prompt)
    return thinking_contents, contents


def form_prompts(system_prompt, user_prompt, tokenizer):
    # prepare the model input
    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt}
        ]
    else:
        messages = []
    messages.append({"role": "user", "content": user_prompt})
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return text


# @torch.no_grad()
# def generate(system_prompt, user_prompt, tokenizer, model, thinking=True, use_vllm=False):
#     # prepare the model input
#     if system_prompt is not None:
#         messages = [
#             {"role": "system", "content": system_prompt}
#         ]
#     else:
#         messages = []
#     messages.append({"role": "user", "content": user_prompt})
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
    
#     if use_vllm:
#         sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32768)
#         outputs = model.generate(prompts=[text], sampling_params=sampling_params)
#         content = outputs[0].outputs[0].text
#         thinking_content = None
#     else:
#         model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#         # conduct text completion
#         generated_ids = model.generate(
#             **model_inputs,
#             max_new_tokens=32768
#         )
#         output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
#         if thinking:
#             # parsing thinking content
#             try:
#                 # rindex finding 151668 (</think>)
#                 index = len(output_ids) - output_ids[::-1].index(151668)
#             except ValueError:
#                 index = 0

#             thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
#             content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
#         else:
#             content = tokenizer.decode(output_ids, skip_special_tokens=True)
#             thinking_content = None
#     return thinking_content, content
    
def load_model(model_name='Qwen/Qwen3-30B-A3B-Thinking-2507'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def load_vllm_model(model_name='Qwen/Qwen3-30B-A3B-Thinking-2507'):
    import os
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(
        model=model_name,
        max_model_len=32768,
        download_dir=os.environ['HF_HOME']
    )
    return llm, tokenizer
    
       
def main():
    # import time
    # start_time = time.time()
    
    # load the domains
    domains = json.load(open(args.domain_path, 'r'))
    # load the model
    logger.info(f'Loading model from {args.model_name}')
    if args.use_vllm:
        model, tokenizer = load_vllm_model(args.model_name)
    else:
        model, tokenizer = load_model(args.model_name)
    
    # load the prompt
    user_prompt_template = open(args.prompt_path.replace("[command]", args.command), 'r').read()
    system_prompt = None
    
    
    if args.command == 'domain2q_woctx' or args.command == 'domain2q_wctx':
        all_prompts = []
        for domain, sub_domains in tqdm(domains.items()):
            for sub_domain in sub_domains:
                user_prompt = user_prompt_template.replace("[domain]", domain).replace("[sub-domain]", sub_domain).replace("[num_questions]", "20")
                all_prompts.append(form_prompts(system_prompt, user_prompt, tokenizer))
            
        thinking_contents, contents = generate(all_prompts, tokenizer, model, thinking=False, use_vllm=args.use_vllm)
        i = 0
        for domain, sub_domains in tqdm(domains.items()):
            for sub_domain in sub_domains:
                write_json(contents[i], Path(args.output_dir) / f'{domain}_{sub_domain}_questions.json')
                i += 1
    elif args.command == 'domains':
        system_prompt = None
        prompts = [form_prompts(system_prompt, user_prompt, tokenizer)]
        thinking_contents, contents = generate(prompts, tokenizer, model, thinking=False, use_vllm=args.use_vllm)
        write_json(contents[0], Path(args.output_dir) / 'domains.json')
    elif args.command == 'q_and_docs':
        system_prompt = None
        user_prompt = user_prompt_template.replace("[length]", str(args.length))
        thinking_content, content = generate(system_prompt, user_prompt, tokenizer, model, thinking=False, use_vllm=args.use_vllm)
        write_json(content, Path(args.output_dir) / f'{domain}_{sub_domain}_q_and_docs.json')
    elif args.command == 'q2docs' or args.command == 'q2docs_1' or args.command == 'q2docs_2' or args.command == 'q2docs_3' or args.command == 'q2docs_4':
        all_prompts = []
        all_questions = []
        # for domain, sub_domains in tqdm(domains.items()):
        for _id in args.domain_ids:
            domain = list(domains.keys())[_id]
            sub_domains = domains[domain]
            for sub_domain in sub_domains:
                if (Path(args.output_dir) / f'{domain}_{sub_domain}_q_and_docs.json').exists():
                    continue
                questions = json.load(open(f'vllm_outputs/questions_{args.question_source}/{domain}_{sub_domain}_questions.json', 'r'))['questions']
                all_questions.append(questions)
                for question in tqdm(questions):
                    system_prompt = None
                    user_prompt = user_prompt_template.replace("[length]", str(args.length)).replace("[Question]", question)
                    all_prompts.append(form_prompts(system_prompt, user_prompt, tokenizer))
                    
        thinking_contents, contents = generate(all_prompts, tokenizer, model, thinking=False, use_vllm=args.use_vllm)
        
        i = 0
        sub_domain_index = 0
        for _id in args.domain_ids:
            domain = list(domains.keys())[_id]
            sub_domains = domains[domain]
            for sub_domain in sub_domains:
                all_data = []
                # loop over all questions for this sub_domain
                for question in all_questions[sub_domain_index]:
                    content = contents[i]
                    i += 1
                    try:
                        data = literal_eval(content)
                    except:
                        print(content)
                        data = {"positive_documents": content}
                    data['question'] = question
                    all_data.append(data)
                sub_domain_index += 1
                write_json(all_data, Path(args.output_dir) / f'{domain}_{sub_domain}_q_and_docs.json')
    elif args.command == 'existing_q2docs' or args.command == 'existing_q2docs_1' or args.command == 'existing_q2docs_2':
        assert args.question_source == 'existing'
        data = read_jsonl('../data/eli5+researchy_questions_1k.jsonl')
        questions = [inst['question'] for inst in data]
        all_data = []
        all_prompts = []
        for question in tqdm(questions):
            system_prompt = None
            user_prompt = user_prompt_template.replace("[length]", str(args.length)).replace("[Question]", question)
            all_prompts.append(form_prompts(system_prompt, user_prompt, tokenizer))
            
        thinking_contents, contents = generate(all_prompts, tokenizer, model, thinking=False, use_vllm=args.use_vllm)
            
        for thinking_content, content in zip(thinking_contents, contents):
            try:
                data = literal_eval(content)
            except:
                print('Error parsing content\n', content)
                data = {"positive_documents": content}
            data['question'] = question
            all_data.append(data)
        write_jsonl(all_data, Path(args.output_dir) / 'existing_q2docs_1k.jsonl')
        
                
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain_path', type=str, default='outputs/domains.json')
    parser.add_argument('--use_vllm', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen3-30B-A3B-Instruct-2507')
    parser.add_argument('--prompt_path', type=str, default='prompts/prompt_[command].txt')
    parser.add_argument('--command', type=str, default='domain2q', choices=['domain2q_woctx', 'domain2q_wctx', 'domains', 'q_and_docs', 'q2docs', 'q2docs_1', 'q2docs_2', 'q2docs_3', 'q2docs_4', 'existing_q2docs', 'existing_q2docs_1', 'existing_q2docs_2'])
    parser.add_argument('--question_source', type=str, default='wctx', choices=['wctx', 'woctx', 'existing', 'woctx_20'])
    parser.add_argument('--output_dir', type=str, default='outputs/')
    parser.add_argument('--domain_ids', type=int, nargs='+', default=[0])
    parser.add_argument('--length', type=int, default=200)
    args = parser.parse_args()
    
    main()
    
    # python generate.py --command q2docs_1 --output_dir outputs/q_docs_wctx_1/ --question_source wctx
    # python generate.py --command q2docs_1 --output_dir outputs/q_docs_woctx_1/ --question_source woctx
    
    # python generate.py --command existing_q2docs --output_dir outputs/q_docs_existing/ --question_source existing
    
    
    
    # python generate.py --command q2docs_1 --output_dir outputs/q_docs_woctx_1/ --question_source woctx_20
    # python generate.py --command q2docs_2 --output_dir outputs/q_docs_woctx_2/ --question_source woctx_20
    # python generate.py --command q2docs_3 --output_dir outputs/q_docs_woctx_3/ --question_source woctx_20