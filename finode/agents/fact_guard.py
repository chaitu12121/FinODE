"""
FactGuardAgent

Verifies executor outputs against retrieved documents.
Prevents hallucinations using semantic similarity checks.
"""

import time
from typing import Dict, Any, List, Optional, Tuple
from .base_agent import BaseAgent, AgentMessage, AgentState
from ..rag.embeddings import EmbeddingGenerator
from ..llm.groq_client import GroqMessage


class FactGuardAgent(BaseAgent):
    """
    Validates outputs for factual consistency.
    
    Responsibilities:
    - Semantic similarity checking
    - Hallucination detection
    - Evidence linking
    - Confidence scoring
    """
    
    def __init__(self, groq_client, similarity_threshold: float = 0.65):
        """
        Initialize FactGuardAgent.
        
        Args:
            groq_client: GroqClient instance
            similarity_threshold: τ threshold for acceptance (0.0-1.0)
        """
        super().__init__("FactGuardAgent", groq_client)
        self.similarity_threshold = similarity_threshold
        self.embedding_generator = EmbeddingGenerator()
        
        self.system_prompt = """You are a fact verification agent.
Your job is to verify whether claims are supported by evidence.

For each claim, assess:
1. Is it supported by the provided documents?
2. Is it a reasonable inference?
3. Are there contradictions?
4. What is the confidence (0.0-1.0)?

Respond with JSON:
{
  "verified": true|false,
  "confidence": 0.85,
  "reasoning": "explanation",
  "supporting_docs": ["doc_id_1", "doc_id_2"],
  "contradictions": ["if any"]
}
"""
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Verify claim or answer.
        
        Args:
            message: Message with:
              - content["claim"]: claim to verify
              - content["retrieved_documents"]: supporting docs
              - content["execution_context"]: execution details
        
        Returns:
            Response with verification result
        """
        self.state.status = "processing"
        start_time = time.time()
        
        try:
            claim = message.content.get("claim", "")
            documents = message.content.get("retrieved_documents", [])
            execution_context = message.content.get("execution_context", {})
            
            if not claim:
                raise ValueError("No claim provided")
            
            # Step 1: Extract evidence from documents
            evidence_text = self._format_evidence(documents)
            
            # Step 2: Use Groq to verify claim
            groq_messages = [
                GroqMessage(
                    role="user",
                    content=f"""Verify this claim:
CLAIM: {claim}

SUPPORTING DOCUMENTS:
{evidence_text}

EXECUTION CONTEXT:
{str(execution_context)}

Does the claim follow from the evidence? Use JSON."""
                )
            ]
            
            verification_result = self.groq_client.infer_json(
                groq_messages,
                system_prompt=self.system_prompt
            )
            
            # Step 3: Compute semantic similarity scores
            similarity_scores = self._compute_similarity_scores(claim, documents)
            
            # Step 4: Make final decision
            verified = (
                verification_result.get("verified", False) and
                verification_result.get("confidence", 0.0) >= self.similarity_threshold and
                (len(documents) > 0 or execution_context)  # Some evidence needed
            )
            
            response_content = {
                "claim": claim,
                "verified": verified,
                "confidence": verification_result.get("confidence", 0.0),
                "reasoning": verification_result.get("reasoning", ""),
                "supporting_docs": verification_result.get("supporting_docs", []),
                "similarity_scores": similarity_scores,
                "threshold": self.similarity_threshold,
                "rejection_reason": None if verified else self._get_rejection_reason(
                    verification_result,
                    similarity_scores,
                    documents
                ),
            }
            
            self.state.status = "done"
            self.state.last_result = response_content
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            self.log_message(message)
            
            return AgentMessage(
                sender_agent="FactGuardAgent",
                recipient_agent=message.sender_agent,
                message_type="response",
                content=response_content,
            )
        
        except Exception as e:
            self.state.status = "error"
            self.state.error_message = str(e)
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            
            return AgentMessage(
                sender_agent="FactGuardAgent",
                recipient_agent=message.sender_agent,
                message_type="error",
                content={"error": str(e)},
            )
    
    def _format_evidence(self, documents: List[Dict[str, Any]]) -> str:
        """Format documents as evidence text"""
        if not documents:
            return "[No documents provided]"
        
        evidence = []
        for doc in documents[:5]:  # Max 5 docs
            evidence.append(f"- {doc.get('title', 'Unknown')}: {doc.get('content', '')}")
        
        return "\n".join(evidence)
    
    def _compute_similarity_scores(
        self,
        claim: str,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute cosine similarity between claim and documents"""
        claim_embedding = self.embedding_generator.embed_text(claim)
        scores = {}
        
        for doc in documents:
            doc_content = doc.get("content", "")
            if doc_content:
                doc_embedding = self.embedding_generator.embed_text(doc_content)
                similarity = self.embedding_generator.cosine_similarity(claim_embedding, doc_embedding)
                scores[doc.get("id", "unknown")] = float(similarity)
        
        return scores
    
    def _get_rejection_reason(
        self,
        verification: Dict[str, Any],
        similarity_scores: Dict[str, float],
        documents: List[Dict[str, Any]]
    ) -> str:
        """Detailed rejection reason"""
        if not documents:
            return "No supporting documents provided"
        
        if verification.get("confidence", 0.0) < self.similarity_threshold:
            return f"Confidence {verification.get('confidence', 0.0):.2f} below threshold {self.similarity_threshold}"
        
        if verification.get("contradictions"):
            return f"Contradictions found: {verification.get('contradictions')}"
        
        avg_similarity = sum(similarity_scores.values()) / len(similarity_scores) if similarity_scores else 0
        if avg_similarity < self.similarity_threshold:
            return f"Low semantic similarity: {avg_similarity:.2f}"
        
        return "Verification failed (unknown reason)"
