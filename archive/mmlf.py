mqr_prompt_2 = "You are an AI language model assistant. Your task is to generate exactly two different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Original question: [query]\nFormat your response in plain text as:\nSub-query 1:\nSub-query 2:\n"
mqr_prompt_5 = "You are an AI language model assistant. Your task is to generate exactly five different versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Original question: [query]\nFormat your response in plain text as:\nSub-query 1:\nSub-query 2:\nSub-query 3:\nSub-query 4:\nSub-query 5:\n"

cqe_prompt = "Please write a passage to answer the following user questions simultaneously. Question 1: [original_query] Question 2: [sub_query]\nFormat your response in plain text as:\nPassage:\n"


"""
Simple vLLM inference script for Llama-370B-Instruct model.
"""
from re import sub
from vllm import LLM, SamplingParams
import argparse
import json
import time
def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def format_prompt_mqr(prompt, question):
    return prompt.replace("[query]", question)


def format_prompt_cqe(prompt, original_query, sub_query):
    return prompt.replace("[original_query]", original_query).replace("[sub_query]", sub_query)

def main():
    parser = argparse.ArgumentParser(description="vLLM inference with Llama-3-70B-Instruct")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        help="Model name or path"
    )
    parser.add_argument("--data_type", type=str, default="ambiguous_qe", choices=["ambiguous_qe", "qampari"])
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio"
    )
    parser.add_argument(
        "--command", type=str, default="generate_subqueries", choices=["generate_subqueries", "generate_passage"]
    )
    
    args = parser.parse_args()
    
    
    # Initialize the LLM
    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        trust_remote_code=True
    )
    
    # Set sampling parameters: temperature=1, top_p=1
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_tokens
    )
    
    if args.command == "generate_subqueries":
        if args.data_type == "ambiguous_qe":
            prompt = mqr_prompt_2
            questions = [inst['question'] for inst in read_jsonl("data/questions/ambiguous_qe_dev_question_only_2_to_5_ctxs.jsonl")]
        elif args.data_type == "qampari":
            prompt = mqr_prompt_5
            questions = [inst['question_text'] for inst in read_jsonl("data/questions/qampari_dev_question_only_5_to_8_ctxs.jsonl")]
            
        prompts = [format_prompt_mqr(prompt, question) for question in questions]
    elif args.command == "generate_passage":
        prompt = cqe_prompt
        subqueries_list = read_jsonl(f"data/mmlf/{args.data_type}_subqueries.jsonl")
        if args.data_type == "ambiguous_qe":
            questions = [inst['question'] for inst in read_jsonl("data/questions/ambiguous_qe_dev_question_only_2_to_5_ctxs.jsonl")]
        elif args.data_type == "qampari":
            questions = [inst['question_text'] for inst in read_jsonl("data/questions/qampari_dev_question_only_5_to_8_ctxs.jsonl")]
            
        len_subqueries = []
        prompts = []
        for subqueries, question in zip(subqueries_list, questions):
            len_subqueries.append(len(subqueries))
            for subquery in subqueries:
                prompts.append(format_prompt_cqe(prompt, question, subquery))
        assert sum(len_subqueries) == len(prompts), (sum(len_subqueries), len(prompts))
    else:
        raise ValueError(f"Invalid command: {args.command}")
    
    # Run inference
    print(f"\nExample Input prompt: {prompts[0]}\n")
    print("Generating response...\n")
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    subqueries = []
    passages = []
    invalid_query_1 = 0
    invalid_query_2 = 0
    # Process the generated text
    for output in outputs:
        generated_text = output.outputs[0].text
        # print("=" * 80)
        # print("Generated response:")
        # print("=" * 80)
        # print(generated_text)
        # print("=" * 80)
        if args.command == "generate_subqueries":
            if args.data_type == "ambiguous_qe":
                if len(generated_text.split("Sub-query 1:")) > 1:
                    sub_query_1 = generated_text.split("Sub-query 1:")[1].strip('\n').split("\n")[0]
                    if len(generated_text.split("Sub-query 2:")) > 1:
                        sub_query_2 = generated_text.split("Sub-query 2:")[1].strip('\n').split("\n")[0]
                    else:
                        sub_query_2 = generated_text.split("Sub-query 1:")[1].strip('\n').split("\n")[1]
                else:
                    sub_query_1 = ""
                    invalid_query_1 += 1
                    if len(generated_text.split("Sub-query 2:")) > 1:
                        sub_query_2 = generated_text.split("Sub-query 2:")[1].strip('\n').split("\n")[0]
                    else:
                        sub_query_2 = ""
                        invalid_query_2 += 1
                if sub_query_1 == "" or sub_query_2 == "":
                    print("=" * 80)
                    print("Generated response:")
                    print("=" * 80)
                    print(generated_text)
                    print("=" * 80)
                    print(f"Sub-query 1: {sub_query_1}")
                    print(f"Sub-query 2: {sub_query_2}")
                    print("=" * 80)
                subqueries.append([sub_query_1, sub_query_2])
            elif args.data_type == "qampari":
                if len(generated_text.split("Sub-query 1:")) > 1:
                    sub_query_5 = generated_text.split("Sub-query 5:")[1].strip('\n').split("\n")[0]
                    sub_query_4 = generated_text.split("Sub-query 4:")[1].strip('\n').split("\n")[0]
                    sub_query_3 = generated_text.split("Sub-query 3:")[1].strip('\n').split("\n")[0]
                    sub_query_2 = generated_text.split("Sub-query 2:")[1].strip('\n').split("\n")[0]
                    sub_query_1 = generated_text.split("Sub-query 1:")[1].strip('\n').split("\n")[0]
                    print(f"Sub-query 1: {sub_query_1}")
                    print(f"Sub-query 2: {sub_query_2}")
                    print(f"Sub-query 3: {sub_query_3}")
                    print(f"Sub-query 4: {sub_query_4}")
                    print(f"Sub-query 5: {sub_query_5}")
                else:
                    sub_query_1 = ""
                    invalid_query_1 += 1
                    sub_query_2 = ""
                    sub_query_3 = ""
                    sub_query_4 = ""
                    sub_query_5 = ""
                    invalid_query_2 += 1
                    
                subqueries.append([sub_query_1, sub_query_2, sub_query_3, sub_query_4, sub_query_5])
            
        elif args.command == "generate_passage":
            print("=" * 80)
            print("Generated response:")
            print("=" * 80)
            print(generated_text)
            print("=" * 80)
            passages.append({"question": generated_text.split("Passage:")[-1].strip('\n').split("\n")[0],
                             "answers": [''],
                             "ctxs": []})
    
    if args.command == "generate_subqueries":
        with open(f"data/mmlf/{args.data_type}_subqueries.jsonl", "w") as f:
            for subqueries in subqueries:
                f.write(json.dumps(subqueries) + "\n")
        print(f"Invalid query 1: {invalid_query_1}")
        print(f"Invalid query 2: {invalid_query_2}")
    elif args.command == "generate_passage":
        with open(f"data/mmlf/{args.data_type}_passages.jsonl", "w") as f:
            for passages in passages:
                f.write(json.dumps(passages) + "\n")
        with open(f"data/mmlf/{args.data_type}_lens.txt", "w") as f:
            for len_subqueries in len_subqueries:
                f.write(str(len_subqueries) + "\n")  
    
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
if __name__ == "__main__":
    main()
