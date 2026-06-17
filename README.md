# Semantic Search

A semantic search application built using LangChain, Ollama, and ChromaDB to perform meaning-based document retrieval using vector embeddings.

## Features

* Semantic document search
* Vector database powered by ChromaDB
* Local embeddings and LLM support through Ollama
* Document chunking and preprocessing
* Fast similarity-based retrieval
* Fully local and privacy-friendly setup

## Tech Stack

* LangChain
* Ollama
* ChromaDB

## Installation

1. Clone the repository:

```bash
git clone https://github.com/devesh-saini/Semantic-Search.git
cd semantic-search
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
```

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Requirements

```text
chromadb
langchain_chroma
langchain_ollama
langchain_community
langchain_text_splitters
```

## Running the Project

Run the main application:

```bash
python main.py
```

> Replace `main.py` with your project's actual entry-point file if different.

## How It Works

1. Load documents into the system.
2. Split documents into smaller chunks.
3. Generate embeddings using Ollama.
4. Store embeddings in ChromaDB.
5. Search using natural language queries.
6. Retrieve the most semantically relevant results.

## Example Query

```text
"What are the key responsibilities of a machine learning engineer?"
```

The system returns documents or passages that are semantically related to the query, even if they do not contain the exact keywords.

## License

This project is licensed under the MIT License.
