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

from trl import SFTTrainer
import csv
from datasets import Dataset, DatasetDict
import random
import locale

from huggingface_hub import login
login("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")


# login("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")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print("DEVICE:- ",DEVICE)

def read_data(file_path):
  data =[]
  with open(file_path, mode='r', encoding='utf-8') as file:
      tsv_reader = csv.reader(file, delimiter='\t')
      header = next(tsv_reader)
      for row in tsv_reader:
        instruction = "In this task, you will receive sentences containing two masked variables. Your objective is to analyze the context provided by the sentence and determine whether a relation exists between the two masked entities. Begin your response with 'True' if a relation exists or 'False' if no relation exists, followed by a brief justification for your decision. This approach will help in understanding how well you can discern and validate relationships based on the given context."
        data.append({"instruction" : instruction, "sentence":row[0],"response": row[1]})
  return data


def balanced_split(data, val_ratio=0.2):
    true_data = [d for d in data if d['response'] == 'True']
    false_data = [d for d in data if d['response'] == 'False']

    # Calculate validation size for each category
    val_size_true = int(len(true_data) * val_ratio)
    val_size_false = int(len(false_data) * val_ratio)

    # Generate random indices for validation sets
    val_indices_true = set(random.sample(range(len(true_data)), val_size_true))
    val_indices_false = set(random.sample(range(len(false_data)), val_size_false))

    # Create balanced training and validation sets
    train_set = [true_data[i] for i in range(len(true_data)) if i not in val_indices_true] + \
                [false_data[i] for i in range(len(false_data)) if i not in val_indices_false]
    val_set = [true_data[i] for i in val_indices_true] + [false_data[i] for i in val_indices_false]

    # Shuffle the sets to mix True and False entries
    random.shuffle(train_set)
    random.shuffle(val_set)

    return train_set, val_set



def generate_training_prompt(
    conversation: str, summary: str, system_prompt: str
) -> str:
    return f"""### Instruction: {system_prompt}

### Input Sentence:
{conversation.strip()}

### Output:
{summary}
""".strip()


def pick_balanced_data(data, count=5):
    true_items = [item for item in data if item['response'] == 'True']
    false_items = [item for item in data if item['response'] == 'False']

    # Ensure there are enough items to pick from
    if len(true_items) < count or len(false_items) < count:
        raise ValueError("Not enough data to pick from.")

    # Randomly select 'count' items from each list
    selected_true_items = random.sample(true_items, count)
    selected_false_items = random.sample(false_items, count)

    # Combine and shuffle the selected items
    selected_items = selected_true_items + selected_false_items
    random.shuffle(selected_items)

    return selected_items

def generate_text_feature(data):
    return [{'text': generate_training_prompt(item['sentence'], item['response'], item['instruction'])} for item in data]


def create_model_and_tokenizer(MODEL_NAME):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map={"":0})
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, add_eos_token=True)

    return model, tokenizer

import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Your script description here")
    parser.add_argument("--train_samples", type=int, default=3000, help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=300, help="Number of validation samples")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--train_file_path", type=str, default="./train_data.tsv", help="Path to the training file")
    parser.add_argument("--output_dir", type=str, default="./experiments/3000_5_FT_model", help="Output directory")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Name or path of the model")

    args = parser.parse_args()

    train_samples = args.train_samples
    val_samples = args.val_samples
    n_epochs = args.n_epochs
    batch_size = args.batch_size

    # Replace 'your_file.tsv' with the path to your actual file
    train_file_path = args.train_file_path
    OUTPUT_DIR = args.output_dir #f"./experiments/{train_samples}_{n_epochs}_FT_model"
    MODEL_NAME = args.model_name#"meta-llama/Llama-2-7b-hf"


    train_data = read_data(train_file_path)
    # test_data = read_data(test_file_path)
    print("Read Data!! ")
    train_set, val_set = balanced_split(train_data)
    # train_set = pick_balanced_data(train_set, count=train_samples)
    # val_set = pick_balanced_data(val_set, count=val_samples)



    train_dataset = generate_text_feature(train_set)
    val_dataset = generate_text_feature(val_set)


    # Assuming train_data and val_data are your lists of dictionaries like the sample provided
    train_dataset = Dataset.from_dict({'text': [item['text'] for item in train_dataset]})
    val_dataset = Dataset.from_dict({'text': [item['text'] for item in val_dataset]})

    # Combine them into a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
    print("Dataset dictionary: \n",dataset_dict)



    model, tokenizer = create_model_and_tokenizer(MODEL_NAME)
    model.config.use_cache = False

    print(model.config.quantization_config.to_dict())

    lora_r = 16
    lora_alpha = 64
    lora_dropout = 0.1
    lora_target_modules = [
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj",
    ]


    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_arguments = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        logging_steps=1,
        learning_rate=1e-4,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=n_epochs,
        evaluation_strategy="steps",
        eval_steps=0.2,
        warmup_ratio=0.05,
        save_strategy="epoch",
        group_by_length=True,
        # report_to="tensorboard",
        output_dir= OUTPUT_DIR,
        save_safetensors=True,
        lr_scheduler_type="cosine",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=4096,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    trainer.train()
    trainer.save_model()

    print(trainer.model)


    # trained_model = AutoPeftModelForCausalLM.from_pretrained(
    #     OUTPUT_DIR,
    #     low_cpu_mem_usage=True,
    # )

    # merged_model = model.merge_and_unload()
    # trained_model.save_pretrained(f"./experiments/{train_samples}_{n_epochs}_merged_model", safe_serialization=True)
    # tokenizer.save_pretrained(f"./experiments/{train_samples}_{n_epochs}_merged_model")

