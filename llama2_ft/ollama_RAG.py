import requests
import json
import re
import csv
import pandas as pd
import torch
from datasets import Dataset, load_dataset
import ollama
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ast
import logging

# Configure logging
logging.basicConfig(filename='ollama_predictions.log', level=logging.INFO, format='%(asctime)s: %(message)s', filemode='a')

# Test logging
logging.info('Starting prediction logging')




def read_data(file_path):
  data =[]
  with open(file_path, mode='r', encoding='utf-8') as file:
      tsv_reader = csv.reader(file, delimiter='\t')
      header = next(tsv_reader)
      # Check if the header actually contains column names, otherwise consider it as data
      if header[0].lower() != 'sentence' or header[1].lower() != 'label':
          data.append({
              "text_input": "You will be given a sentence. Your job is to return a boolean variable True/False indicating if there is a relation between the two masked variables based on the words in the sentence around the masked variables. \n sentence: {} \n response: {}".format(header[0], header[1])
          })

      for row in tsv_reader:
        instruction = """For each given sentence, analyze the context to determine if there is a relationship between the two masked variables indicated by "@GENE$" and "@DISEASE$". Your response should be in the format of a dictionary with two keys: "Answer" and "Explanation". The "Answer" key should map to either True or False - nothing other than that, indicating the presence or absence of a relationship. The "Explanation" key should provide a brief rationale for your decision. 
        For example, your response should look like this: 
        {
            Answer: True,
            Explanation: "The gene mutation is linked to the progression of the disease according to the sentence."
        }. 
        
        Please provide your analysis based on the sentence given below: \n  """

        # instruction = "You will be given a sentence. Your job is to return a boolean variable True/False indicating if there is a relation between the two masked variables."
        data.append({"instruction" : instruction, "sentence":row[0],"response": row[1]})
  return data


def test_ollama_response(data):
    # Construct the input for Ollama based on the provided data
    ollama_input = f"{data['instruction']}\n\n{data['sentence']}"

    # Simulate sending the input to the Ollama API and obtaining a response
    response = ollama.chat(
        model='adrienbrault/biomistral-7b:Q2_K',
        messages=[{'role': 'user', 'content': ollama_input}],
        # stream=True,
    )

    # Assuming we only need the first response from the stream
    # response = next(stream)

    # Extract the content of the response
    prediction = response['message']['content']

    # logging.info(f"Ollama's Prediction: {prediction}")
    # logging.info(f"Actual Response: {data['response']}")
    # logging.info(prediction)
    return prediction



def process_prediction(prediction_string):
    try:
        # prediction_dict = ast.literal_eval(prediction_string)
        answer = prediction_dict['Answer']
        explanation = prediction_dict['Explanation']
        # logging.info(f"Answer: {answer}")
        # logging.info(f"Explanation: {explanation}")
        return answer
    except ValueError as e:
        logging.info(f"Error parsing the dictionary: {e}")
        return 


def evaluate_predictions(predictions, labels):
    # Convert labels to boolean values for compatibility with sklearn metrics
    labels = [True if label.lower() == 'true' else False for label in labels]

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")



# Replace 'your_file.tsv' with the path to your actual file
train_file_path = './train_data.tsv'
test_file_path = './test_data.tsv'



train_samples = 3000
val_samples = 300
n_epochs = 5
batch_size = 32


train_data = read_data(train_file_path)
test_data = read_data(test_file_path)




# Example of processing test data and evaluating predictions
test_predictions = []
test_labels=[]
# Simulate obtaining predictions for each item in test_data
for item in test_data[:5]:
    # Assuming 'get_prediction_from_model' is a function that sends the item to your model and gets a prediction
    raw_prediction = test_ollama_response(item)
    processed_prediction = process_prediction(raw_prediction)
    if processed_prediction:
        logging.info(item["sentence"])
        logging.info(item['response'],processed_prediction )
        test_predictions.append(processed_prediction)
        test_labels.append(item['response'])
# Extract the actual labels from the test data
# test_labels = [item['response'] for item in test_data[:50]]

# Evaluate the predictions against the actual labels
evaluate_predictions(test_predictions, test_labels)
