#import asyncio
#from langchain_core.documents import Document
import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
#from langchain_core.vectorstores import InMemoryVectorStore


####
## Loading PDF.
####

file_path = "./example_data/docker_cheatsheet.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()


####
## Splitting Text.
####

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True
)

all_splits = text_splitter.split_documents(docs)

#print(len(all_splits))



####
## Making Embeddings.
####

embeddings = OllamaEmbeddings(model="mistral:instruct")

vector1 = embeddings.embed_query(all_splits[0].page_content)
vector2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector1) == len(vector2)
print(f"Generated vectors of length: {len(vector2)}")
#print(vector1[:10])



####
## Chroma DB.
####

chroma_client = chromadb.Client()
if "Docker" not in [c.name for c in chroma_client.list_collections()]:
    collection = chroma_client.create_collection("Docker")
else:
    collection = chroma_client.get_collection("Docker")

vector_store = Chroma(
    collection_name="Docker",
    embedding_function=embeddings,
    persist_directory="./docker_langchain_vector_db"
)


####
## Adding Docs to "Nike" collection.
####

#ids = vector_store.add_documents(all_splits)

userQuery = input("=> ")
print("-------------------")
results = vector_store.similarity_search(
            userQuery
        )

print(results[0])
print("-------------------")
