"""
Multi-Hop Retrieval

Performs semantic search with multiple hops to expand context.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from .embeddings import EmbeddingGenerator


class MultiHopRetriever:
    """
    Multi-hop retrieval system for expanding document context.
    
    Algorithm:
    1. Embed query
    2. Find top-k documents
    3. For each round (hop):
       - Expand query with retrieved document content
       - Find new documents
       - Remove duplicates
    """
    
    def __init__(self, embedding_generator: EmbeddingGenerator, groq_client):
        """
        Initialize multi-hop retriever.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            groq_client: GroqClient for query expansion
        """
        self.embedding_generator = embedding_generator
        self.groq_client = groq_client
    
    def retrieve(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        num_hops: int = 2,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with multi-hop expansion.
        
        Args:
            query: Original user query
            documents: List of all available documents
            num_hops: Number of retrieval hops
            top_k: Top-k documents per hop
        
        Returns:
            Ranked list of retrieved documents with scores
        """
        
        # Embed all documents (in production, use vector DB)
        doc_embeddings = {}
        for doc in documents:
            content = doc.get("content", "") or ""
            doc_embeddings[doc["id"]] = self.embedding_generator.embed_text(content)
        
        # Initialize with original query
        current_query = query
        retrieved_ids = set()
        all_results = {}
        
        # Multi-hop loop
        for hop in range(num_hops):
            # Embed current query
            query_embedding = self.embedding_generator.embed_text(current_query)
            
            # Score all documents
            hop_scores = {}
            for doc in documents:
                if doc["id"] not in retrieved_ids:
                    similarity = self.embedding_generator.cosine_similarity(
                        query_embedding,
                        doc_embeddings[doc["id"]]
                    )
                    hop_scores[doc["id"]] = similarity
            
            # Get top-k for this hop
            sorted_docs = sorted(hop_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            for doc_id, score in sorted_docs:
                if doc_id not in all_results:
                    doc = next(d for d in documents if d["id"] == doc_id)
                    all_results[doc_id] = {
                        **doc,
                        "relevance_score": score,
                        "first_hop": hop,
                    }
                    retrieved_ids.add(doc_id)
            
            # Expand query for next hop
            if hop < num_hops - 1 and all_results:
                top_doc_id = sorted_docs[0][0]
                top_doc = next(d for d in documents if d["id"] == top_doc_id)
                current_query = f"{query}. Related context: {top_doc.get('content', '')[:200]}"
        
        # Return sorted by relevance
        results = sorted(
            all_results.values(),
            key=lambda x: x["relevance_score"],
            reverse=True
        )
        
        return results
