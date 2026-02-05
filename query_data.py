"""
Enhanced RAG Query System
Features: 
- Smart Query Router with Pydantic + JSON Output Parser
- MMR retrieval for diverse results
- Hybrid search (semantic + keyword)
- Conversation history support
- Better prompting with source citations
- Configurable retrieval parameters
"""

import argparse
from typing import List, Tuple, Optional, Literal
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_classic.schema import Document
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from enum import Enum
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
CHROMA_PATH = "chroma"

# Retrieval Configuration
DEFAULT_K = 5  # Number of documents to retrieve
DEFAULT_SCORE_THRESHOLD = 0.5  # Minimum relevance score (lowered for better recall)
MMR_FETCH_K = 20  # Number of documents to fetch before MMR filtering
MMR_LAMBDA = 0.7  # Balance between relevance (1) and diversity (0)


# ============== PYDANTIC MODEL FOR QUERY CLASSIFICATION ==============
class QueryClassification(BaseModel):
    """Pydantic model for structured query classification output."""
    query_type: Literal["general", "knowledge"] = Field(
        description="Type of query: 'general' for greetings/chitchat, 'knowledge' for document questions"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )


# Query classification prompt with JSON format instructions
ROUTER_PROMPT = """You are a query classifier. Analyze the user's query and classify it.

Categories:
1. "general" - Greetings, chitchat, simple questions that don't need external documents
   Examples: "Hi", "Hello", "How are you?", "What's your name?", "Thanks!", "Bye", "What can you do?"

2. "knowledge" - Questions that require searching documents/knowledge base for answers
   Examples: "What is the title of the book?", "Who is Alice?", "Explain the plot", "What happens in chapter 1?"

Query: {query}

{format_instructions}
"""

# General conversation prompt (no retrieval needed)
GENERAL_PROMPT = """You are a friendly document assistant. The user is having a conversation with you.

Conversation History:
{chat_history}

Current Question: {question}

Respond naturally and remember the conversation context. If they greet you, greet them back. 
If they share information about themselves (like their name), acknowledge and remember it."""

# Enhanced prompt template with better instructions
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.
Your goal is to give accurate, comprehensive answers using ONLY the information in the context.

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Quote relevant passages when helpful
4. Be specific and detailed in your answers
5. If multiple pieces of context are relevant, synthesize them

CONTEXT:
{context}

---

QUESTION: {question}

ANSWER:"""

# Conversation-aware prompt template
CONVERSATION_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based on the provided context.
Your goal is to give accurate, comprehensive answers using ONLY the information in the context.
You also have access to the conversation history to understand follow-up questions.

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Use conversation history to understand the context of follow-up questions
3. If the context doesn't contain enough information, say so clearly
4. Quote relevant passages when helpful
5. Be specific and detailed in your answers

CONTEXT:
{context}

---

CONVERSATION HISTORY:
{chat_history}

---

CURRENT QUESTION: {question}

ANSWER:"""


class RAGQueryEngine:
    """Enhanced RAG Query Engine with Query Router and multiple retrieval strategies."""
    
    def __init__(
        self, 
        chroma_path: str = CHROMA_PATH,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3
    ):
        """Initialize the RAG engine."""
        self.embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        self.db = Chroma(
            persist_directory=chroma_path, 
            embedding_function=self.embedding_function
        )
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.chat_history: List[Tuple[str, str]] = []
        
        # Initialize the query router chain with JSON parser
        self._init_router_chain()
        
        logger.info(f"Initialized RAG engine with {self.db._collection.count()} documents")
    
    def _init_router_chain(self):
        """Initialize the query classification chain with Pydantic + JSON parser."""
        # Create JSON output parser with Pydantic model
        self.json_parser = JsonOutputParser(pydantic_object=QueryClassification)
        
        # Create router prompt with format instructions
        router_prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
        
        # Build the classification chain: Prompt -> LLM -> JSON Parser
        self.router_chain = router_prompt | self.llm | self.json_parser
        
        # General response chain (no JSON needed here)
        general_prompt = ChatPromptTemplate.from_template(GENERAL_PROMPT)
        self.general_chain = general_prompt | self.llm
    
    def classify_query(self, query: str) -> dict:
        """
        Classify the query using Pydantic model + JSON output parser.
        
        Returns:
            dict with keys: query_type, confidence, reasoning
        """
        try:
            # Get format instructions from the parser
            format_instructions = self.json_parser.get_format_instructions()
            
            # Invoke the chain
            result = self.router_chain.invoke({
                "query": query,
                "format_instructions": format_instructions
            })
            
            # Result is already parsed as dict by JsonOutputParser
            logger.info(f"Query classified as: {result['query_type'].upper()} "
                       f"(confidence: {result['confidence']:.2f})")
            logger.info(f"Reasoning: {result['reasoning']}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Classification failed, defaulting to KNOWLEDGE: {e}")
            return {
                "query_type": "knowledge",
                "confidence": 0.5,
                "reasoning": "Classification failed, defaulting to knowledge query"
            }
    
    def handle_general_query(self, question: str, classification: dict) -> dict:
        """Handle general queries without retrieval but with conversation memory."""
        # Format chat history for the prompt
        history_text = ""
        if self.chat_history:
            history_text = "\n".join([
                f"User: {q}\nAssistant: {a}" 
                for q, a in self.chat_history[-3:]  # Last 3 exchanges
            ])
        else:
            history_text = "(No previous conversation)"
        
        # Invoke with conversation history
        response = self.general_chain.invoke({
            "question": question,
            "chat_history": history_text
        })
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Update chat history
        self.chat_history.append((question, response_text))
        
        return {
            "response": response_text,
            "sources": "",
            "documents": [],
            "query_type": "general",
            "classification": classification,
            "context_used": ""
        }
    
    def retrieve_with_mmr(
        self, 
        query: str, 
        k: int = DEFAULT_K,
        fetch_k: int = MMR_FETCH_K,
        lambda_mult: float = MMR_LAMBDA
    ) -> List[Document]:
        """
        Retrieve documents using Maximal Marginal Relevance (MMR).
        MMR balances relevance with diversity to avoid redundant results.
        """
        results = self.db.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )
        logger.info(f"MMR retrieved {len(results)} documents")
        return results
    
    def retrieve_with_scores(
        self, 
        query: str, 
        k: int = DEFAULT_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with relevance scores.
        Filters results below the score threshold.
        """
        results = self.db.similarity_search_with_relevance_scores(query, k=k)
        
        # Filter by score threshold
        filtered_results = [
            (doc, score) for doc, score in results 
            if score >= score_threshold
        ]
        
        logger.info(f"Retrieved {len(filtered_results)}/{len(results)} documents above threshold {score_threshold}")
        
        # Log scores for debugging
        for doc, score in filtered_results:
            logger.debug(f"Score: {score:.3f} - {doc.page_content[:50]}...")
        
        return filtered_results
    
    def hybrid_retrieve(
        self, 
        query: str, 
        k: int = DEFAULT_K
    ) -> List[Document]:
        """
        Hybrid retrieval combining MMR for diverse semantic results.
        For even better hybrid search, you could add BM25 keyword matching.
        """
        # Get MMR results for diversity
        mmr_results = self.retrieve_with_mmr(query, k=k)
        
        # Get similarity results with scores for relevance ranking
        sim_results = self.retrieve_with_scores(query, k=k)
        
        # Combine and deduplicate
        seen_content = set()
        combined = []
        
        # Prioritize high-scoring similarity results
        for doc, score in sim_results:
            content_key = doc.page_content[:100]
            if content_key not in seen_content:
                seen_content.add(content_key)
                combined.append(doc)
        
        # Add MMR results for diversity
        for doc in mmr_results:
            content_key = doc.page_content[:100]
            if content_key not in seen_content and len(combined) < k * 2:
                seen_content.add(content_key)
                combined.append(doc)
        
        return combined[:k]
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string with source info."""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            file_name = doc.metadata.get("file_name", source)
            page = doc.metadata.get("page", "")
            page_info = f" (Page {page})" if page else ""
            
            context_parts.append(
                f"[Source {i}: {file_name}{page_info}]\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def format_sources(self, documents: List[Document]) -> str:
        """Format source citations."""
        sources = []
        seen = set()
        
        for doc in documents:
            file_name = doc.metadata.get("file_name", doc.metadata.get("source", "Unknown"))
            page = doc.metadata.get("page", "")
            
            source_key = f"{file_name}:{page}" if page else file_name
            if source_key not in seen:
                seen.add(source_key)
                page_info = f" (Page {page})" if page else ""
                sources.append(f"• {file_name}{page_info}")
        
        return "\n".join(sources)
    
    def query(
        self, 
        question: str, 
        use_mmr: bool = True,
        k: int = DEFAULT_K,
        use_history: bool = False,
        skip_routing: bool = False
    ) -> dict:
        """
        Query the RAG system with smart routing.
        
        Args:
            question: The question to answer
            use_mmr: Use MMR for diverse retrieval (recommended)
            k: Number of documents to retrieve
            use_history: Include conversation history
            skip_routing: Skip query classification and always use RAG
            
        Returns:
            Dictionary with response, sources, and retrieved documents
        """
        # Step 1: Classify the query (unless skipped)
        classification = None
        if not skip_routing:
            classification = self.classify_query(question)
            
            # Handle general queries without retrieval (using if-else based on JSON result)
            if classification["query_type"] == "general":
                return self.handle_general_query(question, classification)
        
        # Step 2: Knowledge query - Retrieve documents
        if use_mmr:
            documents = self.retrieve_with_mmr(question, k=k)
        else:
            results = self.retrieve_with_scores(question, k=k)
            documents = [doc for doc, _ in results]
        
        if not documents:
            return {
                "response": "I couldn't find any relevant information to answer your question.",
                "sources": "",
                "documents": [],
                "query_type": "knowledge",
                "classification": classification,
                "context_used": ""
            }
        
        # Format context
        context_text = self.format_context(documents)
        
        # Build prompt
        if use_history and self.chat_history:
            history_text = "\n".join([
                f"Human: {h}\nAssistant: {a}" 
                for h, a in self.chat_history[-3:]  # Last 3 exchanges
            ])
            prompt_template = ChatPromptTemplate.from_template(CONVERSATION_PROMPT_TEMPLATE)
            prompt = prompt_template.format(
                context=context_text, 
                chat_history=history_text,
                question=question
            )
        else:
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=question)
        
        # Generate response
        logger.info("Generating response...")
        response = self.llm.invoke(prompt)
        response_text = response.content
        
        # Update chat history
        self.chat_history.append((question, response_text))
        
        # Format sources
        sources_text = self.format_sources(documents)
        
        return {
            "response": response_text,
            "sources": sources_text,
            "documents": documents,
            "query_type": "knowledge",
            "classification": classification,
            "context_used": context_text
        }
    
    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []
        logger.info("Conversation history cleared")


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(description="Query your documents using RAG")
    parser.add_argument("query_text", type=str, help="The question to ask")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Number of documents to retrieve")
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR (use simple similarity)")
    parser.add_argument("--show-context", action="store_true", help="Show retrieved context")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model to use")
    args = parser.parse_args()
    
    try:
        # Initialize engine
        engine = RAGQueryEngine(model_name=args.model)
        
        # Query
        result = engine.query(
            question=args.query_text,
            use_mmr=not args.no_mmr,
            k=args.k
        )
        
        # Display results
        print("\n" + "="*60)
        query_type = result.get("query_type", "unknown").upper()
        classification = result.get("classification", {})
        confidence = classification.get("confidence", 0) if classification else 0
        reasoning = classification.get("reasoning", "") if classification else ""
        
        print(f"[Query Type: {query_type} | Confidence: {confidence:.0%}]")
        if reasoning:
            print(f"[Reasoning: {reasoning}]")
        print("-"*60)
        print("ANSWER:")
        print("="*60)
        print(result["response"])
        
        # Only show sources for knowledge queries
        if result.get("sources"):
            print("\n" + "-"*60)
            print("SOURCES:")
            print("-"*60)
            print(result["sources"])
        
        if args.show_context:
            print("\n" + "-"*60)
            print("RETRIEVED CONTEXT:")
            print("-"*60)
            print(result["context_used"])
            
    except Exception as e:
        logger.error(f"Error during query: {e}")
        raise


# Interactive mode for conversation
def interactive_mode():
    """Run in interactive conversation mode."""
    print("\n" + "="*60)
    print("RAG Document Assistant - Interactive Mode")
    print("Type 'quit' to exit, 'clear' to clear history")
    print("="*60 + "\n")
    
    engine = RAGQueryEngine()
    
    while True:
        try:
            question = input("\nYou: ").strip()
            
            if not question:
                continue
            if question.lower() == 'quit':
                print("Goodbye!")
                break
            if question.lower() == 'clear':
                engine.clear_history()
                print("History cleared!")
                continue
            
            result = engine.query(question, use_history=True)
            
            query_type_icon = "💬" if result.get("query_type") == "general" else "📖"
            print(f"\n{query_type_icon} Assistant: {result['response']}")
            
            # Only show sources for knowledge queries
            if result.get("sources"):
                print(f"\n📚 Sources:\n{result['sources']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
