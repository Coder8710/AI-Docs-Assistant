# Langchain RAG Document Assistant

An enhanced RAG (Retrieval Augmented Generation) system for querying documents using LangChain and OpenAI.

## ✨ Features

- **Multi-format Support**: Load PDF, Markdown, Word (DOCX), and TXT files
- **Smart Query Router**: Automatically detects if questions need document retrieval or can be answered directly
- **Conversation Memory**: Maintains chat history across all query types for contextual follow-up questions
- **Smart Chunking**: Document-type aware splitting with optimal chunk sizes
- **MMR Retrieval**: Maximal Marginal Relevance for diverse, relevant results
- **Rich Metadata**: Enhanced document metadata for better filtering and citations
- **Interactive Mode**: Chat interface with persistent conversation history
- **Better Prompting**: Detailed prompts with source citations
- **Structured Outputs**: Pydantic models with JSON parsing for reliable query classification
- **Configurable**: Adjustable retrieval parameters (k, threshold, model)

## 🛠️ Tech Stack

- **LangChain** - Framework for LLM applications and RAG pipelines
- **OpenAI GPT-4o-mini** - Language model for query classification and answer generation
- **OpenAI Embeddings** (text-embedding-3-small) - Vector embeddings for semantic search
- **ChromaDB** - Vector database for storing and retrieving document embeddings
- **Pydantic** - Data validation and structured output parsing
- **Python 3.12+** - Core programming language
- **Document Loaders**:
  - `PyPDFLoader` - PDF document processing
  - `Docx2txtLoader` - Word document processing
  - `UnstructuredMarkdownLoader` - Markdown processing
  - `TextLoader` - Plain text files

## 📦 Installation

1. **Install dependencies:**

```bash
pip install -r requirements.txt
pip install "unstructured[md]"
```

2. **Set up your OpenAI API key:**

Create a `.env` file in the project root:
```
OPENAI_API_KEY=your-api-key-here
```

> Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

## 🗄️ Create Database

Add your documents (PDF, MD, DOCX, TXT) to the `data/books` folder, then run:

```bash
python create_database.py
```

**What it does:**
- Loads all PDF, Markdown, Word, and TXT files from `data/books`
- Splits documents with smart chunking (preserves semantic structure)
- Enriches metadata (chunk_id, word_count, file type, timestamps)
- Creates embeddings using OpenAI's `text-embedding-3-small` model
- Stores in ChromaDB with cosine similarity

## 🔍 Query the Database

### Single Query Mode
```bash
python query_data.py "How does Alice meet the Mad Hatter?"
```

The system automatically detects if your query needs document retrieval:
- **General queries** ("Hi", "Hello") → Direct response, no retrieval
- **Knowledge queries** → Searches documents and provides answer with sources

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

This starts an interactive session with **full conversation memory**:
- Remembers your name, preferences, and previous questions
- Maintains context across both general and knowledge queries
- Type `clear` to reset conversation history
- Type `quit` to exit

**Example conversation:**
```
You: Hi, my name is John
Assistant: Hello John! Nice to meet you...

You: What's my name?
Assistant: Your name is John!
```

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
├── query_data.py         # RAG query engine with smart routing
├── requirements.txt      # Dependencies
├── .env                  # API keys (DO NOT COMMIT)
├── .env.example          # Template for environment variables
├── data/
│   └── books/           # Put your documents here (PDF, MD, DOCX, TXT)
└── chroma/              # Vector database (auto-generated)
```

## 🎯 How It Works

### Query Router (Pydantic + JSON Parser)
The system uses a smart query classifier that:
1. Analyzes each query with LLM
2. Returns structured JSON with:
   - `query_type`: "general" or "knowledge"
   - `confidence`: 0-1 score
   - `reasoning`: Why this classification
3. Routes to appropriate handler:
   - **General** → Direct response (no retrieval)
   - **Knowledge** → Full RAG pipeline

### Retrieval Strategies

**MMR (Maximal Marginal Relevance) - Default**
Balances relevance with diversity to avoid redundant results. Best for:
- Broad questions
- When you want varied perspectives

**Similarity Search**
Pure relevance-based retrieval. Best for:
- Specific factual questions
- When you need the most relevant chunks
