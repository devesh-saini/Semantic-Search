from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

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

print(len(docs))
