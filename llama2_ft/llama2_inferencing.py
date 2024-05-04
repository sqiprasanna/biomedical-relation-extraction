from llama2_fine_tuning_main import pick_balanced_data,read_data, balanced_split, create_model_and_tokenizer
import json
import re
from pprint import pprint

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from huggingface_hub import notebook_login
from peft import LoraConfig, PeftModel, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
import time
from trl import SFTTrainer
import csv
from datasets import Dataset, DatasetDict
import random
import locale

# login("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")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("DEVICE:- ",DEVICE)


DEFAULT_SYSTEM_PROMPT = """
Below is a sentence with biomedical entities masked as $GENE$ and $DISEASE$ with biomedical related text. Return boolean variable True/False if there is a relation between these two entities (masked variables) based on context. .
""".strip()


def generate_test_prompt(
    conversation: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> str:
    return f"""### Instruction: {system_prompt}

### Input:
{conversation.strip()}

### Response:
""".strip()

def generate_test_feature(data):
    return [{'sentence':item['sentence'],'response':item['response'],'text': generate_test_prompt(item['sentence'],item['instruction'])} for item in data]



# if __init__ == "__main__":
#     train_samples = 3000
#     val_samples = 300
#     n_epochs = 5
#     batch_size = 32
#     test_file_path = './test_data.tsv'
#     OUTPUT_DIR = f"./experiments/{train_samples}_{n_epochs}_FT_model"
    
    
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument("--train_samples", type=int, default=3000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=300, help="Number of validation samples")
    parser.add_argument("--test_samples", type=int, default=300, help="Number of test samples")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    # parser.add_argument("--train_file_path", type=str, default="./train_data.tsv", help="Path to the training file")
    parser.add_argument("--test_file_path", type=str, default="./test_data.tsv", help="Path to the testing file")
    parser.add_argument("--output_dir", type=str, default="./experiments/3000_5_FT_model", help="Output directory")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name or path of the model")
    parser.add_argument("--save_filepath", type=str, default="./test_df_FT_model.csv", help="save output dataframe path")

    args = parser.parse_args()

    train_samples = args.train_samples
    val_samples = args.val_samples
    test_samples = args.test_samples
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    # Replace 'your_file.tsv' with the path to your actual file
    # train_file_path = args.train_file_path
    test_file_path = args.test_file_path
    OUTPUT_DIR = args.output_dir #f"./experiments/{train_samples}_{n_epochs}_FT_model"
    MODEL_NAME = args.model_name#"meta-llama/Llama-2-7b-hf"
    save_filepath = args.save_filepath #f"./test_df_{train_samples}_{n_epochs}_FT_model.csv"

    


    test_data = read_data(test_file_path)

    # balanced_data = pick_balanced_data(test_data, count=test_samples)
    # test_dataset = generate_test_feature(balanced_data)
    test_dataset = generate_test_feature(test_data)
    
    model, tokenizer = create_model_and_tokenizer(MODEL_NAME)
    model.config.use_cache = False
    
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    print(test_dataset[0]['text'])
    def summarize(model, text: str):
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        inputs_length = len(inputs["input_ids"][0])

        with torch.inference_mode():
            start_time = time.time()
            outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.0001)
            end_time = time.time()
            generate_time = end_time - start_time
        # print(f"Tokenizer time: {tokenizer_time:.2f} seconds")
        print(f"Generation time: {generate_time:.2f} seconds")
        # print(outputs)
        return tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

    summaries = []
    new_rows = []
    test_df = pd.DataFrame(columns = ['sentence','actual_output','predicted_output'])
    for i,item in enumerate(test_dataset):
        start_time = time.time()  # Record the start time
        summary = summarize(model, item['text'])
        end_time = time.time()  # Record the end time
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        print(f"Time taken to summarize item {i + 1}: {elapsed_time:.2f} seconds")

        summaries.append(summary.split("\n")[1])
        new_row = {
            'sentence': item['sentence'],
            'actual_output': item['response'],
            'predicted_output': summary.split("\n")[1]
        }
        new_rows.append(new_row)
        print(new_row)
    new_rows_df = pd.DataFrame(new_rows)
    test_df = pd.concat([test_df, new_rows_df], ignore_index=True)

    from sklearn.metrics import precision_score, recall_score, f1_score

    # Assuming 'df' is your DataFrame
    actual = test_df['actual_output']#.astype(int)
    predicted = test_df['predicted_output'].str.replace(r"[^TrueFalse]", "", regex=True)#.astype(int)
    precision = precision_score(actual, predicted,average='weighted')
    recall = recall_score(actual, predicted,average='weighted')
    f1 = f1_score(actual, predicted,average='weighted')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    test_df.to_csv(save_filepath, index=False)