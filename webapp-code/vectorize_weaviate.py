# Read the content of your file
file_path = r'C:\Users\sai19\Desktop\SJSU\sem4\295-Project\295B\biored_passages.txt'  # Adjust this to the path of your file
with open(file_path, 'r') as file:
    text_data = file.read().splitlines() 


from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert text data to vectors
vectors = [model.encode(text) for text in text_data]

import weaviate

# client = weaviate.Client("https://biomed-0xzvdtpb.weaviate.network")  # Replace with your actual Weaviate URL
client = weaviate.Client(
    url="https://biored-q4r59no1.weaviate.network",
    auth_client_secret=weaviate.AuthApiKey(api_key="rMHGhP2B7cppPjIICoDeW4T10gsygCNdBdFB")
    )
schema = {
    "classes": [
        {
            "class": "BioREDDataset",  # Class names cannot contain spaces; consider using CamelCase or underscores
            "description": "A class to store BioRED Dataset information",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["string"],
                    "description": "The title of the document",
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "The content of the document",
                },
                {
                    "name": "vector",
                    "dataType": ["number[]"],
                    "description": "The vector representation of the document",
                }
            ],
        }
    ]
}

response = client.schema.create(schema)

for text, vector in zip(text_data, vectors):
    data_object = {
        "title": "BioRED data",  
        "content": text,
        "vector": vector.tolist()
    }
    client.data_object.create(class_name= "BioREDDataset", data_object=data_object) 
