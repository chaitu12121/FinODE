"""
Mock demo for FINODE: runs full pipeline without external Groq or sentence-transformers.

Run with: python finode/demo/run_demo_mock.py
"""
import sys
import os
import json
import hashlib
import numpy as np

# Ensure project root is on sys.path so `import finode` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from finode.llm.groq_client import GroqMessage
from finode.agents.planner import PlannerAgent
from finode.agents.supervisor import SupervisorAgent
from finode.agents.executor import ExecutorAgent
from finode.agents.fact_guard import FactGuardAgent
from finode.agents.temporal_gnn_ode import TemporalReasoningAgent
from finode.agents.privacy import PrivacyGovernorAgent
from finode.agents.audit import AuditExplainabilityAgent
from finode.agents.base_agent import AgentMessage


class MockGroqClient:
    """Deterministic mock of GroqClient for offline demo."""
    def __init__(self):
        self.model = "mock-model"
        self.api_key = "mock-key"
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def infer(self, messages, system_prompt=None, temperature=None, max_tokens=None):
        # Not used by demo; provide simple wrapper
        self.call_count += 1
        content = """Mock response"""
        class R:
            pass
        r = R()
        r.choices = [type("C", (), {"message": type("M", (), {"content": content}), "finish_reason": "stop"})()]
        r.usage = type("U", (), {"prompt_tokens": 1, "completion_tokens": 1})()
        return r

    def infer_json(self, messages, system_prompt=None, schema_hint=None):
        self.call_count += 1
        text = "\n".join([m.content for m in messages])

        # Planner prompt
        if "Create a plan for:" in text or "Create a plan for" in text:
            plan = {
                "plan_id": "plan_mock_001",
                "user_query": text.split("Create a plan for:")[-1].strip() if "Create a plan for:" in text else text[:80],
                "intent": "financial_analysis",
                "confidence": 0.98,
                "tasks": [
                    {"task_id": "t1", "description": "Retrieve relevant documents", "agent_type": "RetrievalAgent", "required_inputs": ["query"], "outputs": ["documents"], "dependencies": [], "parameters": {}},
                    {"task_id": "t2", "description": "Run numerical analysis", "agent_type": "ExecutorAgent", "required_inputs": ["documents"], "outputs": ["analysis"], "dependencies": ["t1"], "parameters": {}},
                    {"task_id": "t3", "description": "Verify claims", "agent_type": "FactGuardAgent", "required_inputs": ["analysis", "documents"], "outputs": ["verification"], "dependencies": ["t2"], "parameters": {}},
                ],
                "execution_order": ["t1", "t2", "t3"],
                "success_criteria": ["verification.verified == true"]
            }
            return plan

        # Executor tool selection prompt
        if "Which tools should I use?" in text or "Task:" in text:
            # Return a tool call to calculate compound interest as example
            tool_calls = [
                {
                    "tool_name": "calculate_compound_interest",
                    "parameters": {"principal": 100.0, "rate": 7.0, "periods": 5, "compounds_per_year": 1}
                }
            ]
            return {"tool_calls": tool_calls, "reasoning": "Compound interest computed for projection."}

        # Fact verification prompt
        if "Verify this claim:" in text or "Verify this claim" in text:
            return {"verified": True, "confidence": 0.92, "reasoning": "Claim directly supported by documents.", "supporting_docs": ["doc_1"]}

        # Default
        return {"response": "mock"}

    def batch_infer(self, batch):
        return [self.infer(messages, system) for messages, system in batch]

    def get_stats(self):
        return {"total_calls": self.call_count, "total_input_tokens": self.total_input_tokens, "total_output_tokens": self.total_output_tokens, "model": self.model}


class MockEmbeddingGenerator:
    """Simple deterministic embedding generator not requiring sentence-transformers."""
    def __init__(self, model_name=None, dim: int = 16):
        self.model_name = model_name or "mock-embed"
        self.embedding_dim = dim

    def _text_to_vec(self, text: str):
        h = hashlib.sha256(text.encode()).digest()
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        vec = arr[:self.embedding_dim]
        vec = vec + 1.0
        return vec

    def embed_text(self, text: str):
        return self._text_to_vec(text)

    def embed_batch(self, texts):
        return np.vstack([self._text_to_vec(t) for t in texts])

    def cosine_similarity(self, vec1, vec2):
        v1 = vec1 / (np.linalg.norm(vec1) + 1e-8)
        v2 = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(v1, v2))


def run_mock_demo():
    print('\n--- FINODE Mock Demo (offline) ---\n')

    groq = MockGroqClient()

    # Instantiate agents with mock groq
    planner = PlannerAgent(groq)
    # Lightweight mock retrieval agent to avoid heavy embedding deps
    class MockRetrievalAgent:
        def __init__(self, groq_client):
            self.agent_name = "RetrievalAgent"
            self.state = type("S", (), {"execution_time_ms": 0.0})()
            self.groq_client = groq_client
            # demo documents
            self.documents = [
                {"id": "doc_1", "title": "Apple Q3 2024 Earnings", "content": "Apple Inc. reported Q3 2024 revenue of $93.7 billion, representing 4.7% year-over-year growth.", "metadata": {}},
                {"id": "doc_2", "title": "Tesla Battery Innovation", "content": "Tesla announced a breakthrough in solid-state battery technology with 50% higher energy density.", "metadata": {}},
                {"id": "doc_3", "title": "Federal Reserve Rate Decision", "content": "Federal Reserve raised interest rates by 0.25%.", "metadata": {}},
            ]

        def process_message(self, message: AgentMessage):
            content = {
                "query": message.content.get("query", ""),
                "num_documents_retrieved": len(self.documents),
                "documents": self.documents,
                "retrieval_stats": {"method": "mock"},
            }
            return type("R", (), {"message_type": "response", "content": content})()

    retrieval = MockRetrievalAgent(groq)
    executor = ExecutorAgent(groq)

    # Lightweight mock fact guard to avoid embedding dependency
    class MockFactGuardAgent:
        def __init__(self, groq_client):
            self.agent_name = "FactGuardAgent"
            self.state = type("S", (), {"execution_time_ms": 0.0})()
            self.groq_client = groq_client
            self.embedding_generator = MockEmbeddingGenerator()

        def process_message(self, message: AgentMessage):
            claim = message.content.get("claim", "")
            documents = message.content.get("retrieved_documents", [])
            # Use groq mock to verify
            groq_msg = [GroqMessage(role="user", content=f"Verify this claim: {claim}")]
            verification = self.groq_client.infer_json(groq_msg)
            # compute simple similarity scores
            scores = {}
            claim_vec = self.embedding_generator.embed_text(claim)
            for d in documents:
                scores[d.get('id','doc_unknown')] = self.embedding_generator.cosine_similarity(claim_vec, self.embedding_generator.embed_text(d.get('content','')))
            verification["similarity_scores"] = scores
            return type("R", (), {"message_type": "response", "content": verification})()

    fact_guard = MockFactGuardAgent(groq)
    temporal = TemporalReasoningAgent(groq)
    privacy = PrivacyGovernorAgent(groq)
    audit = AuditExplainabilityAgent(groq)

    # Override embedding generators to avoid external deps
    mock_embed = MockEmbeddingGenerator()
    retrieval.embedding_generator = mock_embed
    fact_guard.embedding_generator = mock_embed

    agents = {
        "PlannerAgent": planner,
        "RetrievalAgent": retrieval,
        "ExecutorAgent": executor,
        "FactGuardAgent": fact_guard,
        "TemporalReasoningAgent": temporal,
        "PrivacyGovernorAgent": privacy,
        "AuditExplainabilityAgent": audit,
    }

    supervisor = SupervisorAgent(groq, agents)

    # Sample query
    query = "What is Apple's recent financial performance and how does it compare to market trends?"
    print(f"Query: {query}\n")

    # Planner
    planner_msg = AgentMessage(sender_agent="Demo", recipient_agent="PlannerAgent", message_type="query", content={"query": query})
    plan_resp = planner.process_message(planner_msg)
    plan = plan_resp.content if plan_resp.message_type == "response" else {}
    print('Plan:', json.dumps(plan, indent=2)[:800], '\n')

    # Retrieval
    retrieval_msg = AgentMessage(sender_agent="Demo", recipient_agent="RetrievalAgent", message_type="query", content={"query": query})
    retrieval_resp = retrieval.process_message(retrieval_msg)
    docs = retrieval_resp.content.get('documents', []) if retrieval_resp.message_type == 'response' else []
    print(f"Retrieved {len(docs)} documents (mock).\n")

    # Supervisor (execute plan)
    supervisor_msg = AgentMessage(sender_agent="Demo", recipient_agent="SupervisorAgent", message_type="query", content={"plan": plan})
    exec_resp = supervisor.process_message(supervisor_msg)
    print('Execution summary:', json.dumps(exec_resp.content, indent=2)[:1000], '\n')

    # Extract final answer
    # Use supervisor results and docs to assemble answer
    if docs:
        final_answer = docs[0].get('content', 'Apple had positive performance')
    else:
        final_answer = "Apple Inc. reported Q3 2024 revenue of $93.7 billion, representing 4.7% YoY growth."

    # Fact verification
    fact_msg = AgentMessage(sender_agent="Demo", recipient_agent="FactGuardAgent", message_type="query", content={"claim": final_answer, "retrieved_documents": docs, "execution_context": exec_resp.content})
    fact_resp = fact_guard.process_message(fact_msg)

    verification = fact_resp.content if fact_resp.message_type == 'response' else {"verified": False, "confidence": 0.0}

    # Privacy
    privacy_msg = AgentMessage(sender_agent="Demo", recipient_agent="PrivacyGovernorAgent", message_type="query", content={"user_role": "analyst", "operation": "aggregate", "data": {"aggregate": final_answer, "trend": "available", "public": "yes"}})
    privacy_resp = privacy.process_message(privacy_msg)

    # Audit
    audit_msg = AgentMessage(sender_agent="Demo", recipient_agent="AuditExplainabilityAgent", message_type="query", content={"event_type":"query_execution","decision":{"answer":final_answer,"verified":verification.get('verified', False),"confidence":verification.get('confidence',0.0)},"features":{"query_length":len(query),"retrieved_docs":len(docs)},"reasoning":"mock run","agent_chain":["PlannerAgent","RetrievalAgent","ExecutorAgent","FactGuardAgent"]})
    audit_resp = audit.process_message(audit_msg)
    audit_entry = audit_resp.content.get('audit_entry', {}) if audit_resp.message_type == 'response' else {}

    # Final JSON
    response = {
        "query": query,
        "answer": final_answer,
        "confidence": float(verification.get('confidence', 0.0)),
        "verified": verification.get('verified', False),
        "evidence": [d.get('title','') for d in docs[:3]],
        "audit_hash": audit_entry.get('hash',''),
        "execution_details": {
            "tasks_executed": exec_resp.content.get('tasks_executed', 0),
            "tasks_failed": exec_resp.content.get('tasks_failed', 0)
        },
        "llm_stats": groq.get_stats(),
    }

    print('\n--- Final Response (mock) ---\n')
    print(json.dumps(response, indent=2))


if __name__ == '__main__':
    run_mock_demo()
