"""
FINODE Real Demo - Fully Functional with Real Groq API

This runs the complete agentic system with actual Groq inference.

Run with: python finode/demo/run_demo_real.py

Requires:
  - GROQ_API_KEY environment variable set
  - pip install groq sentence-transformers cryptography fastapi uvicorn pydantic
"""

import sys
import os
import json
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from finode.llm.groq_client import GroqClient, GroqMessage
from finode.agents.planner import PlannerAgent
from finode.agents.supervisor import SupervisorAgent
from finode.agents.retrieval import RetrievalAgent
from finode.agents.executor import ExecutorAgent
from finode.agents.fact_guard import FactGuardAgent
from finode.agents.temporal_gnn_ode import TemporalReasoningAgent
from finode.agents.privacy import PrivacyGovernorAgent
from finode.agents.audit import AuditExplainabilityAgent
from finode.agents.base_agent import AgentMessage


def print_section(title: str):
    """Print section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_step(step_num: int, title: str):
    """Print step header"""
    print(f"\n[Step {step_num}] {title}")
    print("-" * 80)


def run_real_demo():
    """Run FINODE with real Groq API"""
    
    print_section("FINODE - Financial Intelligence via Neural ODEs - REAL DEMO")
    
    # Initialize Groq client
    print_step(0, "Initialize Groq Client")
    try:
        groq_client = GroqClient()
        print(f"✓ Groq client initialized")
        print(f"  Model: {groq_client.model}")
        print(f"  API Key present: {'✓' if groq_client.api_key else '✗'}")
    except Exception as e:
        print(f"✗ Failed to initialize Groq: {e}")
        print("\n  To fix: Set GROQ_API_KEY environment variable")
        print("  Windows: setx GROQ_API_KEY \"your_api_key_here\"")
        print("  Linux/Mac: export GROQ_API_KEY=\"your_api_key_here\"")
        return
    
    # Initialize agents
    print_step(1, "Initialize Agents")
    try:
        agents = {
            "PlannerAgent": PlannerAgent(groq_client),
            "RetrievalAgent": RetrievalAgent(groq_client),
            "ExecutorAgent": ExecutorAgent(groq_client),
            "FactGuardAgent": FactGuardAgent(groq_client),
            "TemporalReasoningAgent": TemporalReasoningAgent(groq_client),
            "PrivacyGovernorAgent": PrivacyGovernorAgent(groq_client),
            "AuditExplainabilityAgent": AuditExplainabilityAgent(groq_client),
        }
        
        for agent_name in agents:
            print(f"  ✓ {agent_name}")
        
        supervisor = SupervisorAgent(groq_client, agents)
        print(f"  ✓ SupervisorAgent")
        
    except Exception as e:
        print(f"✗ Failed to initialize agents: {e}")
        return
    
    # Sample queries
    queries = [
        "What is Apple's recent financial performance?",
        "How can I calculate the compound interest on $1000 at 7% annual rate for 5 years?",
    ]
    
    for query_idx, query in enumerate(queries, 1):
        print_section(f"Query {query_idx}: {query}")
        
        # Phase 1: Planning
        print_step(1, "Phase 1: Create Execution Plan")
        try:
            planner_msg = AgentMessage(
                sender_agent="Demo",
                recipient_agent="PlannerAgent",
                message_type="query",
                content={"query": query}
            )
            
            plan_response = agents["PlannerAgent"].process_message(planner_msg)
            
            if plan_response.message_type == "error":
                print(f"✗ Planning failed: {plan_response.content}")
                continue
            
            plan = plan_response.content
            print(f"✓ Plan created")
            print(f"  Plan ID: {plan.get('plan_id', 'unknown')}")
            print(f"  Tasks: {len(plan.get('tasks', []))}")
            for task in plan.get('tasks', [])[:3]:
                print(f"    - {task.get('task_id')}: {task.get('description')}")
            if len(plan.get('tasks', [])) > 3:
                print(f"    ... and {len(plan.get('tasks', [])) - 3} more")
            
        except Exception as e:
            print(f"✗ Planning error: {e}")
            continue
        
        # Phase 2: Retrieval
        print_step(2, "Phase 2: Retrieve Documents")
        try:
            retrieval_msg = AgentMessage(
                sender_agent="Demo",
                recipient_agent="RetrievalAgent",
                message_type="query",
                content={"query": query}
            )
            
            retrieval_response = agents["RetrievalAgent"].process_message(retrieval_msg)
            
            if retrieval_response.message_type == "error":
                print(f"✗ Retrieval failed")
                documents = []
            else:
                documents = retrieval_response.content.get("documents", [])
                print(f"✓ Retrieved {len(documents)} documents")
                
                for doc in documents[:2]:
                    print(f"\n  Document: {doc.get('title')}")
                    content = doc.get('content', '')
                    print(f"  Content: {content[:120]}...")
                    print(f"  Relevance: {doc.get('relevance_score', 0):.2%}")
                
                if len(documents) > 2:
                    print(f"\n  ... and {len(documents) - 2} more documents")
        
        except Exception as e:
            print(f"✗ Retrieval error: {e}")
            documents = []
        
        # Phase 3: Execution via Supervisor
        print_step(3, "Phase 3: Execute Tasks (Supervisor)")
        try:
            supervisor_msg = AgentMessage(
                sender_agent="Demo",
                recipient_agent="SupervisorAgent",
                message_type="query",
                content={"plan": plan}
            )
            
            supervisor_start = time.time()
            execution_response = supervisor.process_message(supervisor_msg)
            supervisor_time = time.time() - supervisor_start
            
            if execution_response.message_type == "error":
                print(f"✗ Execution failed: {execution_response.content}")
                execution_results = {}
            else:
                exec_result = execution_response.content
                print(f"✓ Execution complete")
                print(f"  Tasks executed: {exec_result.get('tasks_executed', 0)}")
                print(f"  Tasks failed: {exec_result.get('tasks_failed', 0)}")
                print(f"  Execution time: {supervisor_time:.2f}s")
                print(f"  Status: {exec_result.get('execution_status', 'unknown')}")
                execution_results = exec_result
        
        except Exception as e:
            print(f"✗ Execution error: {e}")
            execution_results = {}
        
        # Phase 4: Extract final answer
        # Use documents or execution results
        if documents:
            final_answer = documents[0].get('content', 'No answer available')
        else:
            final_answer = "Unable to retrieve answer"
        
        # Phase 5: Fact Verification
        print_step(4, "Phase 4: Fact Verification")
        try:
            fact_msg = AgentMessage(
                sender_agent="Demo",
                recipient_agent="FactGuardAgent",
                message_type="query",
                content={
                    "claim": final_answer,
                    "retrieved_documents": documents,
                    "execution_context": execution_results,
                }
            )
            
            fact_response = agents["FactGuardAgent"].process_message(fact_msg)
            
            if fact_response.message_type == "error":
                print(f"✗ Verification failed")
                verified = False
                confidence = 0.0
            else:
                result = fact_response.content
                verified = result.get("verified", False)
                confidence = result.get("confidence", 0.0)
                
                status = "✓ VERIFIED" if verified else "✗ REJECTED"
                print(f"{status}")
                print(f"  Confidence: {confidence:.0%}")
                print(f"  Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
                
                if not verified and result.get("rejection_reason"):
                    print(f"  Rejection: {result.get('rejection_reason')}")
        
        except Exception as e:
            print(f"✗ Verification error: {e}")
            verified = False
            confidence = 0.0
        
        # Phase 6: Privacy
        print_step(5, "Phase 5: Apply Privacy Controls")
        try:
            privacy_msg = AgentMessage(
                sender_agent="Demo",
                recipient_agent="PrivacyGovernorAgent",
                message_type="query",
                content={
                    "user_role": "analyst",
                    "operation": "aggregate",
                    "data": {"aggregate": final_answer, "trend": "available"}
                }
            )
            
            privacy_response = agents["PrivacyGovernorAgent"].process_message(privacy_msg)
            
            if privacy_response.message_type != "error":
                result = privacy_response.content
                print(f"✓ Privacy applied")
                print(f"  Operation: {result.get('operation')}")
                print(f"  Privacy level: {result.get('privacy_level')}")
        
        except Exception as e:
            print(f"✗ Privacy error: {e}")
        
        # Phase 7: Audit Trail
        print_step(6, "Phase 6: Create Audit Trail")
        try:
            audit_msg = AgentMessage(
                sender_agent="Demo",
                recipient_agent="AuditExplainabilityAgent",
                message_type="query",
                content={
                    "event_type": "query_execution",
                    "decision": {
                        "answer": final_answer,
                        "verified": verified,
                        "confidence": confidence,
                    },
                    "features": {"query_length": len(query), "docs_retrieved": len(documents)},
                    "reasoning": "Multi-agent analysis executed",
                    "agent_chain": ["Planner", "Retrieval", "Executor", "FactGuard"],
                }
            )
            
            audit_response = agents["AuditExplainabilityAgent"].process_message(audit_msg)
            
            if audit_response.message_type != "error":
                result = audit_response.content
                audit_entry = result.get("audit_entry", {})
                print(f"✓ Audit trail generated")
                print(f"  Entry ID: {audit_entry.get('entry_id')}")
                print(f"  Hash: {audit_entry.get('hash', '')[:16]}...")
        
        except Exception as e:
            print(f"✗ Audit error: {e}")
        
        # Final summary
        print_step(7, "Final Response")
        print(f"\nQuery: {query}\n")
        print(f"Answer: {final_answer[:200]}...")
        print(f"\nVerified: {'YES' if verified else 'NO'}")
        print(f"Confidence: {confidence:.0%}")
        print(f"Evidence sources: {len(documents)}")
    
    # System statistics
    print_section("System Statistics")
    stats = groq_client.get_stats()
    print(f"Total LLM calls: {stats['total_calls']}")
    print(f"Total input tokens: {stats['total_input_tokens']}")
    print(f"Total output tokens: {stats['total_output_tokens']}")
    print(f"Model: {stats['model']}")
    
    print("\nAgent execution summaries:")
    for agent_name, agent in agents.items():
        state = agent.get_state()
        print(f"  {agent_name}: {state['status']} ({state['execution_time_ms']:.1f}ms)")
    
    print_section("Demo Complete ✓")
    print("FINODE system executed successfully with real Groq API.\n")


if __name__ == "__main__":
    run_real_demo()
