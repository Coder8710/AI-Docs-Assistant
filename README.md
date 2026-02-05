# Langchain RAG Document Assistant

An enhanced RAG (Retrieval Augmented Generation) system for querying documents using LangChain and OpenAI.

## ✨ Features

- **Multi-format Support**: Load PDF, Markdown, and TXT files
- **Smart Chunking**: Document-type aware splitting with optimal chunk sizes
- **MMR Retrieval**: Maximal Marginal Relevance for diverse, relevant results
- **Rich Metadata**: Enhanced document metadata for better filtering and citations
- **Conversation Mode**: Interactive chat with history for follow-up questions
- **Better Prompting**: Detailed prompts with source citations
- **Configurable**: Adjustable retrieval parameters (k, threshold, model)

## 📦 Install Dependencies

1. Do the following before installing the dependencies found in `requirements.txt` file because of current challenges installing `onnxruntime` through `pip install onnxruntime`. 

    - For MacOS users, a workaround is to first install `onnxruntime` dependency for `chromadb` using:

    ```bash
     conda install onnxruntime -c conda-forge
    ```
    See this [thread](https://github.com/microsoft/onnxruntime/issues/11037) for additional help if needed. 

     - For Windows users, follow the guide [here](https://github.com/bycloudai/InstallVSBuildToolsWindows?tab=readme-ov-file) to install the Microsoft C++ Build Tools. Be sure to follow through to the last step to set the environment variable path.


2. Now run this command to install dependencies in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

3. Install markdown dependencies with: 

```bash
pip install "unstructured[md]"
```

4. Set up your OpenAI API key:

```bash
# Create a .env file with:
OPENAI_API_KEY=your-api-key-here
```

## 🗄️ Create Database

Add your documents (PDF, MD, TXT) to the `data/books` folder, then run:

```bash
python create_database.py
```

**What it does:**
- Loads all PDF, Markdown, and TXT files from `data/books`
- Splits documents with smart chunking (preserves semantic structure)
- Enriches metadata (chunk_id, word_count, file type, timestamps)
- Creates embeddings using OpenAI's `text-embedding-3-small` model
- Stores in ChromaDB with cosine similarity

## 🔍 Query the Database

### Single Query Mode
```bash
python query_data.py "How does Alice meet the Mad Hatter?"
```

### With Options
```bash
# Retrieve more documents
python query_data.py "What happens at the tea party?" --k 7

# Use simple similarity instead of MMR
python query_data.py "Describe the Cheshire Cat" --no-mmr

# Show the retrieved context
python query_data.py "Who is the Queen of Hearts?" --show-context

# Use a different model
python query_data.py "What is the moral of the story?" --model gpt-4o
```

### Interactive Conversation Mode
```bash
python query_data.py --interactive
```

This starts an interactive session where you can:
- Ask follow-up questions (conversation history is maintained)
- Type `clear` to reset history
- Type `quit` to exit

## 🛠️ Configuration

Edit the configuration variables in the Python files:

### create_database.py
```python
CHUNK_SIZE = 500           # Characters per chunk
CHUNK_OVERLAP = 100        # Overlap between chunks
MARKDOWN_CHUNK_SIZE = 800  # Larger chunks for markdown
```

### query_data.py
```python
DEFAULT_K = 5              # Documents to retrieve
DEFAULT_SCORE_THRESHOLD = 0.5  # Minimum relevance score
MMR_FETCH_K = 20           # Candidates for MMR
MMR_LAMBDA = 0.7           # Relevance vs diversity (0-1)
```

## 📁 Project Structure

```
langchain-rag-tutorial/
├── create_database.py    # Document processing & embedding
├── query_data.py         # RAG query engine
├── compare_embeddings.py # Embedding comparison utility
├── requirements.txt      # Dependencies
├── data/
│   └── books/           # Put your documents here (PDF, MD, TXT)
└── chroma/              # Vector database (auto-generated)
```

## 🎯 Retrieval Strategies

### MMR (Maximal Marginal Relevance) - Default
Balances relevance with diversity to avoid redundant results. Best for:
- Broad questions
- When you want varied perspectives

### Similarity Search
Pure relevance-based retrieval. Best for:
- Specific factual questions
- When you need the most relevant chunks

## 📚 Tutorial

Here is a step-by-step tutorial video: [RAG+Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami)
