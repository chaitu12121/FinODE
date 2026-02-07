"""
RetrievalAgent

Multi-hop RAG system with FAISS indexing.
Retrieves cited documents and performs semantic expansion.
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from .base_agent import BaseAgent, AgentMessage, AgentState
from ..rag.embeddings import EmbeddingGenerator
from ..rag.multihop import MultiHopRetriever


class RetrievalAgent(BaseAgent):
    """
    Retrieves relevant documents from knowledge base.
    
    Responsibilities:
    - Query embedding generation
    - Multi-hop semantic search
    - Citation tracking
    - Confidence scoring
    """
    
    def __init__(self, groq_client, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RetrievalAgent.
        
        Args:
            groq_client: GroqClient instance
            embedding_model: Model for embeddings
        """
        super().__init__("RetrievalAgent", groq_client)
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.multihop_retriever = MultiHopRetriever(
            self.embedding_generator,
            groq_client
        )
        
        # In-memory document store (would be replaced with vector DB in production)
        self.documents: Dict[str, Dict[str, Any]] = {}
        self._load_demo_documents()
    
    def _load_demo_documents(self) -> None:
        """Load sample financial documents for demo"""
        demo_docs = [
            {
                "id": "doc_1",
                "title": "Apple Q3 2024 Earnings",
                "content": "Apple Inc. reported Q3 2024 revenue of $93.7 billion, representing 4.7% year-over-year growth.",
                "metadata": {"company": "Apple", "quarter": "Q3 2024", "type": "financial_report"}
            },
            {
                "id": "doc_2",
                "title": "Tesla Battery Innovation",
                "content": "Tesla announced a breakthrough in solid-state battery technology with 50% higher energy density.",
                "metadata": {"company": "Tesla", "date": "2026-01-15", "type": "press_release"}
            },
            {
                "id": "doc_3",
                "title": "Federal Reserve Interest Rate Decision",
                "content": "Federal Reserve raised interest rates by 0.25% to combat inflation pressures.",
                "metadata": {"institution": "Federal Reserve", "date": "2026-01-20", "type": "policy"}
            },
            {
                "id": "doc_4",
                "title": "S&P 500 Index Performance",
                "content": "S&P 500 index hit all-time high at 6,200 points following positive earnings reports.",
                "metadata": {"index": "S&P 500", "date": "2026-02-01", "type": "market_data"}
            },
            {
                "id": "doc_5",
                "title": "Bitcoin Price Surge",
                "content": "Bitcoin price reached $95,000 following SEC approval of Bitcoin spot ETF.",
                "metadata": {"asset": "Bitcoin", "date": "2026-01-25", "type": "crypto_news"}
            },
        ]
        
        for doc in demo_docs:
            self.documents[doc["id"]] = doc
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Retrieve documents for a query.
        
        Args:
            message: Message with content["query"] or content["queries"]
        
        Returns:
            Response with retrieved documents and citations
        """
        self.state.status = "processing"
        start_time = time.time()
        
        try:
            query = message.content.get("query", "")
            if not query:
                raise ValueError("No query provided")
            
            # Perform multi-hop retrieval
            results = self.multihop_retriever.retrieve(
                query=query,
                documents=list(self.documents.values()),
                num_hops=2,
                top_k=5
            )
            
            # Build response with citations
            retrieved_docs = []
            for result in results:
                retrieved_docs.append({
                    "doc_id": result["id"],
                    "title": result["title"],
                    "content": result["content"],
                    "metadata": result["metadata"],
                    "relevance_score": float(result.get("relevance_score", 0.0)),
                    "cite_as": f"{result['title']} (doc_id: {result['id']})",
                })
            
            response_content = {
                "query": query,
                "num_documents_retrieved": len(retrieved_docs),
                "documents": retrieved_docs,
                "retrieval_stats": {
                    "embedding_model": self.embedding_generator.model_name,
                    "num_hops": 2,
                    "top_k": 5,
                }
            }
            
            self.state.status = "done"
            self.state.last_result = response_content
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            self.log_message(message)
            
            return AgentMessage(
                sender_agent="RetrievalAgent",
                recipient_agent=message.sender_agent,
                message_type="response",
                content=response_content,
            )
        
        except Exception as e:
            self.state.status = "error"
            self.state.error_message = str(e)
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            
            return AgentMessage(
                sender_agent="RetrievalAgent",
                recipient_agent=message.sender_agent,
                message_type="error",
                content={"error": str(e)},
            )
    
    def add_document(self, doc_id: str, title: str, content: str, metadata: Dict[str, Any]) -> None:
        """Add new document to knowledge base"""
        self.documents[doc_id] = {
            "id": doc_id,
            "title": title,
            "content": content,
            "metadata": metadata,
        }
