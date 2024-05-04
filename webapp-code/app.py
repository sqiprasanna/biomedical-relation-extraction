from flask import Flask, request, jsonify, send_from_directory, render_template
import openai
import json
import os

import weaviate
import spacy

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from sentence_transformers import SentenceTransformer

import ast

app = Flask(__name__, static_folder='.')

# key = "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"
key = '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'
lm_client = openai.OpenAI(api_key=key)

knowledge = weaviate.Client(
    url="https://test-w6siimpn.weaviate.network",
    additional_headers={"X-OpenAI-Api-Key": key}
)

dictt = {}

custom_functions = [  
    {
        "name": "get_relation",
        "description": "Function to be called when a dictinary needs to be returned",
        "parameters": {
            "type": "object",
            "properties": {
                "relation": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "description": "A dictionary where each key is a string and each value is a list of strings."
                }
            },
            "required": ["relation"]
        }
    }
]


custom_function_ent = [  
    {
        "name": "get_entities",
        "description": "Function be be used when asked to return entities from the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "description": "Return a list of entities"
                }
            },
            "required": ["entities"]
        }
    }
]




custom_function_description = [  
    {
        "name": "get_description",
        "description": "Function to be called when a description needs to be returned",
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "This should be the description being returned",
                },
            },
            "required": ["response"]
        }
    }
]

nlp = spacy.load("en_core_web_sm")

custom_functions_relation = [  
    {
        "name": "get_relation",
        "description": "Function to be called when a list of relations needs to be returned",
        "parameters": {
            "type": "object",
            "properties": {
                "relation": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "description": "A list of relations between the entitites provided."
                }
            },
            "required": ["relation"]
        }
    }
]

def extract_entities(text):
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

def generate_chat_response(user_message):
    system_message = "You will be given a paragraph, and followed by it, a list of entity words extracted from the paragraph, and their descriptions. You must use these words and the paragraph as refrence to create a dictionry, where the keys are the entity words, and value is a list of words, selected from the entity words that are linked to the key word in the sentence. You do not need to use ALL of the entity words as keys within the dictionary. Only the ones that make sense. Return the dictionary."
    
    msg = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = lm_client.chat.completions.create(
        model="gpt-4",
        messages=msg, max_tokens=2000, temperature=0.0,  
        )

    reply = response.choices[0].message.content
    print(reply)

    relation = json.loads(reply)
    print(relation, type(relation))
    print(response)
    if reply is None:
       relation = json.loads(response.choices[0].message.function_call.arguments)["relation"]
    return relation


def generate_chat_response_relationship(user_message):
    system_message = "You will be given pairs of entities, and the original paragrapgh from which they were retrived. You must return a list contaning the relationship between the pairs of entities provided, based on the context from which they were extracted. Furthermore, you will also be given the relationships from which you must pick. In the list, return the relationships in the same order as the pairs of entities provided. I will directly use this list assuming order or output is equal to the order of input. The returned relationships must be ONLY one of the following: \n strr += Relationships to be used:  Positive Correlation,  Negative Correlation, Association, Bind, Drug Interaction, Cotreatment, Comparison, Conversion"
    msg = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = lm_client.chat.completions.create(
        model="gpt-4",
        messages=msg, max_tokens=2000, temperature=0.0,  
        )

    reply = response.choices[0].message.content
    print(reply)
    try:
        reply = ast.literal_eval(reply)
    except:
        pass

    if reply is None:
       reply = json.loads(response.choices[0].message.function_call.arguments)["relation"]

    return reply

def generate_chat_response_entities(inpt):
    system_message = "You will be given a paragrapgh. You must return a list of biomedical entities from the given text. Examples of terms that you kght think are related to biomed, but are nott: 'anti-cancer drug',etc, stuff that is regular english really."
    
    user_message = "Input paragraph:  " + inpt
    

    msg = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = lm_client.chat.completions.create(
        model="gpt-4",
        messages=msg, max_tokens=500, temperature=0.0,  
        functions=custom_function_ent, function_call='auto'
        )

    reply = response.choices[0].message.content

    try:
        reply = ast.literal_eval(reply)
    except:
        pass

    if reply is None:
       reply = json.loads(response.choices[0].message.function_call.arguments)["entities"]

    return reply






def get_description(text, context):
    if text in dictt: return dictt[text]
    system_message = "You will be given a word, and context extracted from a text book. Based off of the information provided in the context, you should return a very short description of the word. The description should be as short as possible. just a couple of words. return an empty string if a description of what the word is cannot be formed from the context provided."    
    user_message = "Word: " + text + "\n\n Context to be used: \n\n" + context
    msg = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = lm_client.chat.completions.create(
        model="gpt-4",
        messages=msg, max_tokens=2000, temperature=0.0,  
        functions=custom_function_description, function_call='auto'
        )

    reply = response.choices[0].message.content
    print(reply)

    try:
       relation = json.loads(response.choices[0].message.function_call.arguments)["response"]
       print(relation)
       return relation
    except:
        return ""
def append_dict_values(input_string, dictionary):
    punctuation = '.,;?!""\''
    words = input_string.split()
    processed_words = set()
    output_string = input_string
    output_string += "\n\nEntity words:\n\n"
    for key,val in dictionary.items():
        output_string += f"\n{key}: {val}."
        processed_words.add(key)
    return output_string, processed_words

def remove_empty_lists(dictionary):
    keys_to_remove = [key for key, value in dictionary.items() if not value]
    for key in keys_to_remove:
        del dictionary[key]
    return dictionary

@app.route('/')
def index():
    return render_template('index.html')

def qdb(query, db_client, name, cname):
    context = None
    metadata = []
    try:
        limit = 4
        res = (
            db_client.query.get(name, ["text", "metadata"])
            .with_near_text({"concepts": query})
            .with_limit(limit)
            .do()
        )
        context = ""
        metadata = []
        chunk_id = 0
        for i in range(limit):
            context += "Chunk ID: " + str(chunk_id) + "\n"
            context += res["data"]["Get"][cname][i]["text"] + "\n\n"
            metadata.append(res["data"]["Get"][cname][i]["metadata"])
            chunk_id += 1
    except Exception as e:
        print("Exception in DB, dude.")
        print(e)
    return context, metadata

def extract_entities_nltk(sentence):
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    nouns = [(word,tag) for word, tag in tagged_tokens if "NN" in tag]
    return nouns

def update_dictt(text):
    print("ENTITTES>>>>>>>>>>>",text)
    for word in text:
        print(word)
        context, _ = qdb(word, knowledge, 'ddbot6', "Ddbot6")
        description = get_description(word, context)
        dictt[word] = description




custom_function_ft = [  
    {
        "name": "get_entities_relation",
        "description": "Function be be used when asked to return relations from the text.",
        "parameters": {
            "type": "object",
            "properties": {
                "entities-relation": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "description": "Return a list of entities with relation between them"
                }
            },
            "required": ["entities-relation"]
        }
    }
]


def generate_chat_response_ft(inpt):
    system_message = "You will be given a paragraph along with list of entities. You must return a list of biomedical entities with relation between them from the given text. "
    
    user_message = "Input paragraph:  " + inpt
    

    msg = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = lm_client.chat.completions.create(
        model="ft:gpt-3.5-turbo-1106:personal:med:90DXyGST",
        messages=msg, max_tokens=500, temperature=0.0
        )

    reply = response.choices[0].message.content
    print("RESPONSE---",reply)
    return reply
import json



@app.route('/process', methods=['POST'])
def process():
    data = request.json
    # process_jsonl_file('path_to_your_file.jsonl')
    sentence = data['sentence']
    entities = generate_chat_response_entities(sentence)
    print(entities, type(entities))
    update_dictt(entities)
    os, pw = append_dict_values(sentence.lower(), dictt)
    print("\n\n\n", os, "\n\n\n")
    pw = list(pw)
    connections = generate_chat_response(os)
    connections = remove_empty_lists(connections)
    print("Getting:\n\n", connections)
    for key, value in connections.items():
        strr = "Original Sentence: " + sentence + "\n"
        # for val in value:
        #     string_to_be_added = str(key) + " & " + str(val)
        #     strr += string_to_be_added
        #     strr += "\n"
        strr+=str(key)
    strr += "Relationships to be used:  Positive Correlation,  Negative Correlation, Association, Bind, Drug Interaction, Cotreatment, Comparison, Conversion"
    # rel = generate_chat_response_relationship(strr)
    print("Input to the model:", strr)
    output = generate_chat_response_ft(strr)
    print("Fine tuned outputs- :", output)
    # print(rel)
    return jsonify({"names": pw, "connections": connections})



    
def process_test(file_path):
    count=0
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            for message in data["messages"]:
                if message["role"] == "user" and count <2:
                    sentence = message["content"]
                    count +=1
                    print(sentence)
                    # sentence = data['sentence']
                    entities = generate_chat_response_entities(sentence)
                    print(entities, type(entities))
                    update_dictt(entities)
                    os, pw = append_dict_values(sentence.lower(), dictt)
                    print("\n\n\n", os, "\n\n\n")
                    pw = list(pw)
                    connections = generate_chat_response(os)
                    connections = remove_empty_lists(connections)
                    print("Getting:\n\n", connections)
                    for key, value in connections.items():
                        strr = "Original Sentence: " + sentence + "\n"
                        strr+=str(key)
                    strr += "Relationships to be used:  Positive Correlation,  Negative Correlation, Association, Bind, Drug Interaction, Cotreatment, Comparison, Conversion"
                    # rel = generate_chat_response_relationship(strr)
                    print("Input to the model:", strr)
                    output = generate_chat_response_ft(strr)
                    print("Fine tuned outputs- :", output)
                    # print(rel)
                    return jsonify({"names": pw, "connections": connections})



if __name__ == '__main__':
    file_path = r'C:\Users\sai19\Desktop\SJSU\sem4\295-Project\processed_bioRED_data_test_100_abs.jsonl'
    process_test(file_path)
    # app.run(debug=True)