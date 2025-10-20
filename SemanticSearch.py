from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

documents = [
    Document(
        page_content="Tony Stark was able to build this in a cave... With a box of scraps.",
        metadata={"source": "Iron-Man-Origin"}
    ),
    Document(
        page_content="Bruce Wayne is the real person behind the mask of Batman.",
        metadata={"source": "Batman-End-of-Masks"}
    )
]

file_path = "./example_data/nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

#print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True
)

all_splits = text_splitter.split_documents(docs)

#print(len(all_splits))

embeddings = OllamaEmbeddings(model="mistral:instruct")

vector1 = embeddings.embed_query(all_splits[0].page_content)
vector2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector1) == len(vector2)
print(f"Generated vectors of length: {len(vector2)}")
#print(vector1[:10])

vector_store = InMemoryVectorStore(embeddings)
ids = vector_store.add_documents(documents = all_splits)
results = await vector_store.asimilarity_search("When was Nike incorporated?")

print(results[0])
