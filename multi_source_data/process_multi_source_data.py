import json

def read_jsonl(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def write_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    data = read_json('Multi_Domain_Query_300.json')
    new_data = []
    for domain_data in data['domains']:
        for q in domain_data['queries']:
            new_data.append({
                "question": q['query_text'],
                "qid": q['id'],
                "ctxs": [],
                "answers": [''],
            })
    print(len(new_data))
    write_json('../data/questions/Multi_Domain_Query_300.json', new_data)

if __name__ == '__main__':
    main()