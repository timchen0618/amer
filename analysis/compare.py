import os
from openai import OpenAI
from dotenv import load_dotenv
import json
import re
from tqdm import trange
# Load environment variables from .env file
load_dotenv()

def read_jsonl(file_path: str) -> list:
    """
    Read a JSONL file and return a list of dictionaries.
    """
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

def parse_list_response(response: str) -> str:
    """
    Parse the response to extract which list is more diverse.
    Returns either 'list1' or 'list2'.
    """
    # First try to find JSON in the response
    json_match = re.search(r'\{[^}]+\}', response)
    if json_match:
        try:
            json_data = json.loads(json_match.group())
            if 'more_diverse_list' in json_data:
                return json_data['more_diverse_list']
        except json.JSONDecodeError:
            pass
    
    # # If no JSON found or parsing failed, look for text patterns
    # if re.search(r'\*\*List 1\*\*.*more diverse', response, re.IGNORECASE | re.DOTALL):
    #     return 'list1'
    # elif re.search(r'\*\*List 2\*\*.*more diverse', response, re.IGNORECASE | re.DOTALL):
    #     return 'list2'
    
    # If no clear indication found, return None
    return None

def get_openai_response(prompt: str, system_prompt: str = "You are a helpful assistant.", model: str = "gpt-4-turbo-preview") -> str:
    """
    Get a response from OpenAI API.
    
    Args:
        prompt (str): The prompt to send to the API
        system_prompt (str): The system prompt that sets the context and behavior of the AI
        model (str): The model to use (default: gpt-4-turbo-preview)
    
    Returns:
        str: The response from the API
    """
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Extract and return the response
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    system_prompt = open("system_prompt.txt", "r").read()
    model = "gpt-4o-2024-08-06"
    
    result_1 = read_jsonl("/datastor1/hungting/retrieval_outputs/mteb_retriever/inf/ambignq+nqopen-all_multi_answer_evidence_dev_2_to_5_ctxs.json")    
    result_2 = read_jsonl("results/ambiguous_inf/toy_contrastive_from_stage2_nq_lr1e4_ep20_warmup_0.05/retrieval_out_dev_ambiguous.jsonl")
    assert len(result_1) == len(result_2)
    
    
    win_rate = 0
    for i in trange(len(result_1)):
        list_1 = '\n'.join(['Document ' + str(j) + ':\n' + c['title'] + ' ' + c['text'] for j, c in enumerate(result_1[i]['ctxs'][:10])])
        list_2 = '\n'.join(['Document ' + str(j) + ':\n' + c['title'] + ' ' + c['text'] for j, c in enumerate(result_2[i]['ctxs'][:10])])
        prompt = f"List 1: \n{list_1}\n\nList 2: \n{list_2}\n\nQuery: {result_1[i]['question']}\n\nWhich list is more diverse in content?"
        response = get_openai_response(prompt, system_prompt, model)
        fw = open(f'responses/{i}.txt', 'w')
        fw.write(response)
        fw.close()
        more_diverse_list = parse_list_response(response)
        if more_diverse_list == 'list1':
            win_rate += 1
    
    print(f"Win rate: {100*win_rate / len(result_1)}")
    
