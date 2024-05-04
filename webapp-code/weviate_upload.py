import weaviate
import docx2txt
import os
import glob
import time
from pdfminer.high_level import extract_text
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from io import StringIO
from pdfminer.high_level import extract_text
import PyPDF2

openapi_key = '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'

client = weaviate.Client(
    url="https://pdfqabot-47o5gind.weaviate.network",
    # auth_client_secret=weaviate.AuthApiKey(api_key="84uDEuxvWPgyTuhWp8bolbTPimhh77jJTMi2"),
    additional_headers={"X-OpenAI-Api-Key": openapi_key}
    )



def split_pdf_text_by_page(pdf_path):
    pages=[]
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            
            pages.append(text)
    return pages

def load_documents(directory, glob_patterns):
    documents = []
    for glob_pattern in glob_patterns:
        file_paths = glob.glob(os.path.join(directory, glob_pattern))
        for fp in file_paths:
            try:
                if fp.endswith('.docx'):
                    text = docx2txt.process(fp)
                    pages = [text]  # Treat the whole document as a single "page"
                elif fp.endswith('.pdf'):
                    pages = split_pdf_text_by_page(fp)
                else:
                    print(f"Warning: Unsupported file format for {fp}")
                    continue
                documents.extend([(page, os.path.basename(fp), i+1) for i, page in enumerate(pages)])
            except Exception as e:
                print(f"Warning: The file {fp} could not be processed. Error: {e}")
    return documents


def split_text(text, file_name, chunk_size, chunk_overlap):
    start = 0
    end = chunk_size
    while start < len(text):
        yield (text[start:end], file_name)
        start += (chunk_size - chunk_overlap)
        end = start + chunk_size

def split_documents(documents, chunk_size, chunk_overlap):
    texts = []
    metadata = []
    for doc_text, file_name, page_number in documents:
        for chunk in split_text(doc_text, file_name, chunk_size, chunk_overlap):
            sentence = chunk[0]
            texts.append(sentence)
            metadata.append(str(file_name) + " Pg: " + str(page_number))

    return texts, metadata




print("---STARTED----\n\n")
directory = r'C:\Users\sai19\Desktop\others\244-AKG\RAG_QA_bot\uploads\'
class_name = "resume"


glob_patterns = ["*.docx", "*.pdf"]
documents = load_documents(directory, glob_patterns)

chunk_size = 400
chunk_overlap = 0
texts, metadata = split_documents(documents, chunk_size, chunk_overlap)

print("-----------------------------")
data_objs = [{"text": tx, "metadata": met}  for tx,met in zip(texts, metadata)]
print(len(data_objs))


i = 0

class_obj = {
    "class": class_name,

    "properties": [
        {
            "name": "text",
            "dataType": ["text"],
        },
        {
            "name": "metadata",
            "dataType": ["text"],
        },
    ],

    "vectorizer": "text2vec-openai",

    "moduleConfig": {
        "text2vec-openai": {
            "vectorizeClassName": False,
            "model": "ada",
            "modelVersion": "002",
            "type": "text"
        },
            
    },
}

print("--------*--------**")

client.batch.configure(batch_size=100)
with client.batch as batch:
    for data_obj in data_objs:
        i+= 1
        if i>-1:
            print("--",i)
            batch.add_data_object(data_obj, class_name)
        else:
            print(i)

res = client.query.get(
    class_name,
    ["text", "metadata"])\
    .with_near_text({"concepts": "What do I do with my life??"})\
    .with_limit(10)\
    .do()

print(res)



