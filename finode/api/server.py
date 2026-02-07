"""
FastAPI Server for FINODE

Exposes the agentic AI system as a REST API.
"""

import json
import asyncio
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
import os
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # If python-dotenv is not installed or fails, continue (env vars may be set externally)
    pass

# Import FINODE components
try:
    from finode.llm.groq_client import GroqClient
    from finode.agents.planner import PlannerAgent
    from finode.agents.supervisor import SupervisorAgent
    from finode.agents.retrieval import RetrievalAgent
    from finode.agents.executor import ExecutorAgent
    from finode.agents.fact_guard import FactGuardAgent
    from finode.agents.temporal_gnn_ode import TemporalReasoningAgent
    from finode.agents.privacy import PrivacyGovernorAgent
    from finode.agents.audit import AuditExplainabilityAgent
    from finode.agents.base_agent import AgentMessage
except ImportError as e:
    raise ImportError(f"Failed to import FINODE components: {e}")


# Pydantic models for API
class QueryRequest(BaseModel):
    """User query request"""
    query: str
    user_role: str = "analyst"


class QueryResponse(BaseModel):
    """System response"""
    query: str
    answer: str
    confidence: float
    evidence: List[str]
    audit_hash: str
    execution_time_ms: float


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FINODESystem:
    """Central FINODE system orchestrator"""
    
    def __init__(self, groq_api_key: Optional[str] = None):
        """
        Initialize FINODE system.
        
        Args:
            groq_api_key: Groq API key (defaults to env var)
        """
        # Initialize Groq client
        self.groq_client = GroqClient(api_key=groq_api_key)
        
        # Initialize agents
        self.planner = PlannerAgent(self.groq_client)
        self.retrieval = RetrievalAgent(self.groq_client)
        self.executor = ExecutorAgent(self.groq_client)
        self.fact_guard = FactGuardAgent(self.groq_client)
        self.temporal = TemporalReasoningAgent(self.groq_client)
        self.privacy_governor = PrivacyGovernorAgent(self.groq_client)
        self.audit = AuditExplainabilityAgent(self.groq_client)
        
        # Initialize supervisor with all agents
        self.supervisor = SupervisorAgent(
            self.groq_client,
            agents={
                "PlannerAgent": self.planner,
                "RetrievalAgent": self.retrieval,
                "ExecutorAgent": self.executor,
                "FactGuardAgent": self.fact_guard,
                "TemporalReasoningAgent": self.temporal,
                "PrivacyGovernorAgent": self.privacy_governor,
                "AuditExplainabilityAgent": self.audit,
            }
        )
    
    async def process_query(self, query: str, user_role: str = "analyst") -> Dict[str, Any]:
        """
        Process a user query through the full FINODE pipeline.
        
        Args:
            query: User query
            user_role: User role for access control
        
        Returns:
            Complete response with answer, evidence, and audit trail
        """
        
        # Step 1: Create execution plan
        logger.info(f"[FINODE] Creating plan for query: {query}")
        planner_msg = AgentMessage(
            sender_agent="API",
            recipient_agent="PlannerAgent",
            message_type="query",
            content={"query": query}
        )
        plan_response = self.planner.process_message(planner_msg)
        
        if plan_response.message_type == "error":
            raise Exception(f"Planning failed: {plan_response.content}")
        
        plan = plan_response.content
        logger.info(f"[FINODE] Plan created with {len(plan.get('tasks', []))} tasks")
        
        # Step 2: Retrieve relevant documents
        logger.info("[FINODE] Retrieving documents...")
        retrieval_msg = AgentMessage(
            sender_agent="API",
            recipient_agent="RetrievalAgent",
            message_type="query",
            content={"query": query}
        )
        retrieval_response = self.retrieval.process_message(retrieval_msg)
        
        if retrieval_response.message_type == "error":
            retrieved_docs = []
        else:
            retrieved_docs = retrieval_response.content.get("documents", [])
        
        logger.info(f"[FINODE] Retrieved {len(retrieved_docs)} documents")
        
        # Step 3: Execute tasks
        logger.info("[FINODE] Executing plan...")
        supervisor_msg = AgentMessage(
            sender_agent="API",
            recipient_agent="SupervisorAgent",
            message_type="query",
            content={"plan": plan}
        )
        execution_response = self.supervisor.process_message(supervisor_msg)
        
        if execution_response.message_type == "error":
            raise Exception(f"Execution failed: {execution_response.content}")
        
        execution_results = execution_response.content
        logger.info(f"[FINODE] Executed {execution_results.get('tasks_executed', 0)} tasks")
        
        # Step 4: Extract final answer
        # Find the main result (usually from RetrievalAgent)
        final_answer = self._extract_answer(retrieved_docs, execution_results)
        
        # Step 5: Verify facts
        logger.info("[FINODE] Verifying facts...")
        fact_guard_msg = AgentMessage(
            sender_agent="API",
            recipient_agent="FactGuardAgent",
            message_type="query",
            content={
                "claim": final_answer,
                "retrieved_documents": retrieved_docs,
                "execution_context": execution_results,
            }
        )
        fact_guard_response = self.fact_guard.process_message(fact_guard_msg)
        
        if fact_guard_response.message_type == "error":
            fact_verification = {"verified": False, "confidence": 0.0}
        else:
            fact_verification = fact_guard_response.content
        
        # Step 6: Apply privacy protections
        logger.info(f"[FINODE] Applying privacy for role: {user_role}")
        privacy_msg = AgentMessage(
            sender_agent="API",
            recipient_agent="PrivacyGovernorAgent",
            message_type="query",
            content={
                "user_role": user_role,
                "operation": "aggregate",
                "data": {
                    "aggregate": final_answer,
                    "trend": "information available",
                    "public": "all results public",
                }
            }
        )
        privacy_response = self.privacy_governor.process_message(privacy_msg)
        
        # Step 7: Generate audit trail
        logger.info("[FINODE] Generating audit trail...")
        audit_msg = AgentMessage(
            sender_agent="API",
            recipient_agent="AuditExplainabilityAgent",
            message_type="query",
            content={
                "event_type": "query_execution",
                "decision": {
                    "answer": final_answer,
                    "verified": fact_verification.get("verified", False),
                    "confidence": fact_verification.get("confidence", 0.0),
                },
                "features": {
                    "query_length": len(query),
                    "retrieved_docs": len(retrieved_docs),
                },
                "reasoning": f"Query processed through {execution_results.get('tasks_executed', 0)} tasks",
                "agent_chain": ["PlannerAgent", "RetrievalAgent", "ExecutorAgent", "FactGuardAgent"],
            }
        )
        audit_response = self.audit.process_message(audit_msg)
        
        if audit_response.message_type == "error":
            audit_trail = {}
        else:
            audit_trail = audit_response.content
        
        # Step 8: Build final response
        total_time = (
            self.planner.state.execution_time_ms +
            self.retrieval.state.execution_time_ms +
            self.supervisor.state.execution_time_ms +
            self.fact_guard.state.execution_time_ms
        )
        
        response = {
            "query": query,
            "answer": final_answer,
            "confidence": float(fact_verification.get("confidence", 0.0)),
            "verified": fact_verification.get("verified", False),
            "evidence": [doc.get("cite_as", "") for doc in retrieved_docs[:3]],
            "execution_details": {
                "total_time_ms": total_time,
                "tasks_executed": execution_results.get("tasks_executed", 0),
                "tasks_failed": execution_results.get("tasks_failed", 0),
            },
            "audit_hash": audit_trail.get("audit_entry", {}).get("hash", ""),
            "audit_trail": audit_trail.get("audit_trail", []),
            "llm_stats": self.groq_client.get_stats(),
        }
        
        return response
    
    def _extract_answer(self, retrieved_docs: List[Dict[str, Any]], execution_results: Dict[str, Any]) -> str:
        """Extract final answer from system execution"""
        # Try to get answer from execution results
        results = execution_results.get("results", {})
        for task_id, task_result in results.items():
            if task_result.get("success"):
                result = task_result.get("result", {})
                if "content" in result:
                    return str(result["content"])
        
        # Fallback: combine retrieved documents
        if retrieved_docs:
            content = " ".join([doc.get("content", "") for doc in retrieved_docs[:2]])
            return content[:500] if content else "Unable to generate answer"
        
        return "No answer available"


# Initialize FastAPI app
app = FastAPI(
    title="FINODE - Financial Intelligence via Neural ODEs",
    description="Agentic AI system for financial analysis",
    version="1.0.0"
)

# Configure CORS to allow frontend preflight requests (adjust via ALLOWED_ORIGINS env var)
allowed = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
origins = [o.strip() for o in allowed.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global FINODE system instance
finode_system: Optional[FINODESystem] = None


@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    global finode_system
    logger.info("Initializing FINODE system...")
    finode_system = FINODESystem()
    logger.info("FINODE system ready")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """
    Process a financial query through FINODE.
    
    Args:
        request: QueryRequest with query and optional user_role
    
    Returns:
        QueryResponse with answer, confidence, and evidence
    """
    try:
        result = await finode_system.process_query(
            query=request.query,
            user_role=request.user_role
        )
        
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            confidence=result["confidence"],
            evidence=result["evidence"],
            audit_hash=result["audit_hash"],
            execution_time_ms=result["execution_details"]["total_time_ms"],
        )
    
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system": "FINODE",
        "version": "1.0.0",
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if finode_system is None:
        return {"error": "System not initialized"}
    
    return {
        "groq_stats": finode_system.groq_client.get_stats(),
        "agent_states": {
            agent_name: agent.get_state()
            for agent_name, agent in finode_system.supervisor.agents.items()
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
