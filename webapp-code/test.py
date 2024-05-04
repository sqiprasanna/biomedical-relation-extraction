
import json

def process_sentence(sentence):
    # Your existing processing logic here
    print(f"Processing sentence: {sentence}")
    # Dummy return for illustration
    return {"entities": ["Entity1", "Entity2"], "connections": {"Entity1": "Entity2"}}

def process_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            for message in data["messages"]:
                if message["role"] == "user":
                    content = message["content"]
                    
                    # result = process_sentence(content)
                    # print(f"Result for sentence '{content}': {result}")

process_jsonl_file(r'C:\Users\sai19\Desktop\SJSU\sem4\295-Project\processed_bioRED_data_test_100_abs.jsonl')
