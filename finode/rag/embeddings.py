"""
Embedding Generation

Converts text to vectors for semantic search.
Uses sentence-transformers for compact, efficient embeddings.
"""

import numpy as np
from typing import List, Dict, Any, Optional


class EmbeddingGenerator:
    """
    Generate embeddings for text using sentence-transformers.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Install with: pip install sentence-transformers"
            )
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Text to embed
        
        Returns:
            Numpy array of embeddings (shape: embedding_dim,)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for batch of texts.
        
        Args:
            texts: List of texts to embed
        
        Returns:
            Numpy array of embeddings (shape: len(texts), embedding_dim)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1, vec2: Embedding vectors
        
        Returns:
            Cosine similarity score (0-1)
        """
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))
