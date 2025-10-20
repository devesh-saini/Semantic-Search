from langchain_core.documents import Document

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
