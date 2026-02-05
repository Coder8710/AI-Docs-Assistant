"""
Enhanced Document Database Creator for RAG
Supports: PDF, Markdown, TXT, DOCX (Word) files
Features: Smart chunking, rich metadata, hybrid search preparation
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain_classic.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
import hashlib
from datetime import datetime
from typing import List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

# Chunking Configuration - Optimized for better retrieval
CHUNK_SIZE = 500  # Increased for more context per chunk
CHUNK_OVERLAP = 100  # Overlap to maintain context continuity
MARKDOWN_CHUNK_SIZE = 800  # Larger chunks for markdown (preserves structure better)


def main():
    """Main entry point for database creation."""
    try:
        generate_data_store()
        logger.info("Database creation completed successfully!")
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        raise


def generate_data_store():
    """Generate the vector store from documents."""
    documents = load_documents()
    if not documents:
        logger.warning("No documents found to process!")
        return
    
    chunks = split_text(documents)
    chunks = enrich_metadata(chunks)
    save_to_chroma(chunks)


def load_documents() -> List[Document]:
    """
    Load documents from multiple file types.
    Supports: PDF, Markdown, TXT, DOCX (Word)
    """
    all_documents = []
    
    # Load Markdown files
    try:
        md_loader = DirectoryLoader(
            DATA_PATH, 
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True
        )
        md_docs = md_loader.load()
        logger.info(f"Loaded {len(md_docs)} Markdown documents")
        all_documents.extend(md_docs)
    except Exception as e:
        logger.warning(f"Error loading Markdown files: {e}")
    
    # Load PDF files
    try:
        pdf_loader = DirectoryLoader(
            DATA_PATH, 
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        pdf_docs = pdf_loader.load()
        logger.info(f"Loaded {len(pdf_docs)} PDF documents")
        all_documents.extend(pdf_docs)
    except Exception as e:
        logger.warning(f"Error loading PDF files: {e}")
    
    # Load Word documents (DOCX)
    try:
        docx_loader = DirectoryLoader(
            DATA_PATH, 
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=True
        )
        docx_docs = docx_loader.load()
        logger.info(f"Loaded {len(docx_docs)} Word documents")
        all_documents.extend(docx_docs)
    except Exception as e:
        logger.warning(f"Error loading Word files: {e}")
    
    # Load TXT files
    try:
        txt_loader = DirectoryLoader(
            DATA_PATH, 
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True
        )
        txt_docs = txt_loader.load()
        logger.info(f"Loaded {len(txt_docs)} TXT documents")
        all_documents.extend(txt_docs)
    except Exception as e:
        logger.warning(f"Error loading TXT files: {e}")
    
    logger.info(f"Total documents loaded: {len(all_documents)}")
    return all_documents


def split_text(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks using appropriate splitter based on document type.
    Uses RecursiveCharacterTextSplitter with optimized settings.
    """
    all_chunks = []
    
    for doc in documents:
        source = doc.metadata.get("source", "")
        
        # Use Markdown splitter for .md files
        if source.endswith(".md"):
            text_splitter = MarkdownTextSplitter(
                chunk_size=MARKDOWN_CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        else:
            # Use RecursiveCharacterTextSplitter for other file types
            # Separators are ordered by priority - tries to split on larger semantic units first
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len,
                add_start_index=True,
                separators=[
                    "\n\n",  # Paragraphs
                    "\n",    # Lines
                    ". ",    # Sentences
                    "? ",    # Questions
                    "! ",    # Exclamations
                    "; ",    # Clauses
                    ", ",    # Phrases
                    " ",     # Words
                    ""       # Characters
                ]
            )
        
        chunks = text_splitter.split_documents([doc])
        all_chunks.extend(chunks)
    
    logger.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
    
    # Preview a sample chunk
    if all_chunks:
        sample_chunk = all_chunks[min(10, len(all_chunks)-1)]
        logger.info(f"Sample chunk preview:\n{sample_chunk.page_content[:200]}...")
        logger.info(f"Sample chunk metadata: {sample_chunk.metadata}")
    
    return all_chunks


def enrich_metadata(chunks: List[Document]) -> List[Document]:
    """
    Enrich chunk metadata for better retrieval and filtering.
    Adds: chunk_id, doc_type, word_count, char_count, created_at
    """
    enriched_chunks = []
    
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        
        # Generate unique chunk ID
        content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
        chunk_id = f"{os.path.basename(source)}_{i}_{content_hash}"
        
        # Determine document type
        if source.endswith(".pdf"):
            doc_type = "pdf"
        elif source.endswith(".md"):
            doc_type = "markdown"
        elif source.endswith((".docx", ".doc")):
            doc_type = "word"
        elif source.endswith(".txt"):
            doc_type = "text"
        else:
            doc_type = "unknown"
        
        # Enrich metadata
        chunk.metadata.update({
            "chunk_id": chunk_id,
            "chunk_index": i,
            "doc_type": doc_type,
            "word_count": len(chunk.page_content.split()),
            "char_count": len(chunk.page_content),
            "created_at": datetime.now().isoformat(),
            "file_name": os.path.basename(source),
        })
        
        enriched_chunks.append(chunk)
    
    logger.info(f"Enriched metadata for {len(enriched_chunks)} chunks")
    return enriched_chunks


def save_to_chroma(chunks: List[Document]):
    """
    Save chunks to ChromaDB vector store.
    """
    if not chunks:
        logger.warning("No chunks to save!")
        return
    
    # Clear out the database first
    if os.path.exists(CHROMA_PATH):
        logger.info(f"Removing existing database at {CHROMA_PATH}")
        shutil.rmtree(CHROMA_PATH)

    # Create embeddings
    logger.info("Creating embeddings (this may take a while)...")
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-small"  # Using newer, better embedding model
    )
    
    # Create a new DB from the documents
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function, 
        persist_directory=CHROMA_PATH,
        collection_metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    
    logger.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}")
    
    # Verify the database
    logger.info(f"Database contains {db._collection.count()} documents")


if __name__ == "__main__":
    main()
