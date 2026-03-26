from openai import OpenAI
from tqdm import tqdm

client = OpenAI()

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


def shuffle_list(lst):
    import random
    random.shuffle(lst)
    return lst

import json
def write_jsonl(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')

# response = get_completion(client, open('instruction_adjectives.txt').read())
# write_file = open('adjectives.txt', 'w')
# write_file.write(response)
# write_file.close()
# print(response)

jobs = open('jobs.txt').read().strip('\n').strip().split(',')
names = open('names.txt').read().strip('\n').strip().split(',')
adjectives = open('adjectives.txt').read().strip('\n').strip().split(',')

all_responses = []
for job, name, adjective in tqdm(zip(jobs, names, adjectives)):
    name_string = f'{name}, which is a {adjective} {job}.'
    prompt = open('instruction.txt').read().replace('[PERSON]', name_string)
    response = get_completion(client, prompt)
    try:
        print(response)
        all_responses.append(json.loads(response))
        
    except:
        print('skip...')
    
write_jsonl('responses.jsonl', all_responses)




