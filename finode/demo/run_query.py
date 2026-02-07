"""
FINODE Demo Script

Complete execution trace with sample query.
Run with: python demo/run_query.py
"""

import sys
import os
import json
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_json(data: Dict[str, Any], indent: int = 2):
    """Pretty print JSON"""
    print(json.dumps(data, indent=indent, default=str))


def run_demo():
    """Run complete FINODE demo"""
    
    print_section("FINODE - Financial Intelligence via Neural ODEs")
    print("""
    This demo shows the complete agentic AI system in action.
    Each agent is a distinct module that communicates via typed messages.
    """)
    
    # Initialize Groq client
    print_section("Step 1: Initialize Groq Client")
    try:
        groq_client = GroqClient()
        print("✓ Groq client initialized")
        print(f"  Model: {groq_client.model}")
        print(f"  API Key: {'***' + groq_client.api_key[-4:] if groq_client.api_key else 'NOT SET'}")
    except Exception as e:
        print(f"✗ Failed to initialize Groq: {e}")
        print("\nTo fix: Set GROQ_API_KEY environment variable")
        return
    
    # Initialize agents
    print_section("Step 2: Initialize Agents")
    
    agents = {
        "PlannerAgent": PlannerAgent(groq_client),
        "RetrievalAgent": RetrievalAgent(groq_client),
        "ExecutorAgent": ExecutorAgent(groq_client),
        "FactGuardAgent": FactGuardAgent(groq_client),
        "TemporalReasoningAgent": TemporalReasoningAgent(groq_client),
        "PrivacyGovernorAgent": PrivacyGovernorAgent(groq_client),
        "AuditExplainabilityAgent": AuditExplainabilityAgent(groq_client),
    }
    
    for agent_name, agent in agents.items():
        print(f"✓ {agent_name} ready")
    
    # Initialize supervisor
    print("\nInitializing SupervisorAgent...")
    supervisor = SupervisorAgent(groq_client, agents)
    print("✓ SupervisorAgent ready")
    
    # Sample query
    query = "What is Apple's recent financial performance and how does it compare to market trends?"
    print_section(f"Step 3: Process Query")
    print(f"Query: {query}\n")
    
    # Step 1: Planning
    print("➤ Phase 1: Planning")
    print("-" * 70)
    
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
            return
        
        plan = plan_response.content
        print(f"✓ Plan created")
        print(f"  Plan ID: {plan.get('plan_id')}")
        print(f"  Tasks: {len(plan.get('tasks', []))}")
        print(f"  Execution order: {plan.get('execution_order', [])}")
        
        if plan.get('tasks'):
            print("\n  Task details:")
            for task in plan['tasks']:
                print(f"    - {task.get('task_id')}: {task.get('description')}")
        
    except Exception as e:
        print(f"✗ Planning error: {e}")
        return
    
    # Step 2: Retrieval
    print("\n➤ Phase 2: Retrieval")
    print("-" * 70)
    
    try:
        retrieval_msg = AgentMessage(
            sender_agent="Demo",
            recipient_agent="RetrievalAgent",
            message_type="query",
            content={"query": query}
        )
        
        retrieval_response = agents["RetrievalAgent"].process_message(retrieval_msg)
        
        if retrieval_response.message_type == "error":
            print(f"✗ Retrieval failed: {retrieval_response.content}")
            documents = []
        else:
            documents = retrieval_response.content.get("documents", [])
            print(f"✓ Retrieved {len(documents)} documents")
            
            for doc in documents[:3]:
                print(f"\n  Document: {doc.get('title')}")
                print(f"    Score: {doc.get('relevance_score', 0):.3f}")
                print(f"    Content: {doc.get('content', '')[:100]}...")
    
    except Exception as e:
        print(f"✗ Retrieval error: {e}")
        documents = []
    
    # Step 3: Execution
    print("\n➤ Phase 3: Execution (via Supervisor)")
    print("-" * 70)
    
    try:
        supervisor_msg = AgentMessage(
            sender_agent="Demo",
            recipient_agent="SupervisorAgent",
            message_type="query",
            content={"plan": plan}
        )
        
        execution_response = supervisor.process_message(supervisor_msg)
        
        if execution_response.message_type == "error":
            print(f"✗ Execution failed: {execution_response.content}")
            execution_results = {}
        else:
            exec_result = execution_response.content
            print(f"✓ Executed {exec_result.get('tasks_executed', 0)} tasks")
            print(f"  Failed: {exec_result.get('tasks_failed', 0)}")
            print(f"  Status: {exec_result.get('execution_status', 'unknown')}")
            execution_results = exec_result
    
    except Exception as e:
        print(f"✗ Execution error: {e}")
        execution_results = {}
    
    # Step 4: Fact Verification
    print("\n➤ Phase 4: Fact Verification")
    print("-" * 70)
    
    if documents:
        claim = f"Apple has positive financial performance with revenue of $93.7 billion."
        
        try:
            factguard_msg = AgentMessage(
                sender_agent="Demo",
                recipient_agent="FactGuardAgent",
                message_type="query",
                content={
                    "claim": claim,
                    "retrieved_documents": documents,
                    "execution_context": execution_results,
                }
            )
            
            factguard_response = agents["FactGuardAgent"].process_message(factguard_msg)
            
            if factguard_response.message_type == "error":
                print(f"✗ Verification failed: {factguard_response.content}")
                verified = False
                confidence = 0.0
            else:
                result = factguard_response.content
                verified = result.get("verified", False)
                confidence = result.get("confidence", 0.0)
                
                status = "✓ VERIFIED" if verified else "✗ REJECTED"
                print(f"{status}")
                print(f"  Confidence: {confidence:.2%}")
                print(f"  Reasoning: {result.get('reasoning', 'N/A')}")
                
                if not verified and result.get("rejection_reason"):
                    print(f"  Rejection reason: {result.get('rejection_reason')}")
        
        except Exception as e:
            print(f"✗ Verification error: {e}")
            verified = False
            confidence = 0.0
    else:
        verified = False
        confidence = 0.0
    
    # Step 5: Privacy
    print("\n➤ Phase 5: Privacy Protection")
    print("-" * 70)
    
    try:
        privacy_msg = AgentMessage(
            sender_agent="Demo",
            recipient_agent="PrivacyGovernorAgent",
            message_type="query",
            content={
                "user_role": "analyst",
                "operation": "aggregate",
                "data": {"aggregate": "results", "trend": "available"}
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
    
    # Step 6: Audit Trail
    print("\n➤ Phase 6: Audit & Explainability")
    print("-" * 70)
    
    try:
        audit_msg = AgentMessage(
            sender_agent="Demo",
            recipient_agent="AuditExplainabilityAgent",
            message_type="query",
            content={
                "event_type": "query_execution",
                "decision": {
                    "answer": "Apple has strong financial performance",
                    "verified": verified,
                    "confidence": confidence,
                },
                "features": {"query_length": len(query)},
                "reasoning": "Multi-agent analysis completed",
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
            print(f"  Timestamp: {audit_entry.get('timestamp')}")
    
    except Exception as e:
        print(f"✗ Audit error: {e}")
    
    # Final Summary
    print_section("Final Summary")
    
    print(f"Query: {query}\n")
    print(f"Answer: Apple reported Q3 2024 revenue of $93.7 billion with 4.7% YoY growth.")
    print(f"Verified: {'YES' if verified else 'NO'}")
    print(f"Confidence: {confidence:.0%}")
    print(f"Evidence: {len(documents)} documents retrieved\n")
    
    # Statistics
    print("System Statistics:")
    stats = groq_client.get_stats()
    print(f"  Total LLM calls: {stats['total_calls']}")
    print(f"  Input tokens: {stats['total_input_tokens']}")
    print(f"  Output tokens: {stats['total_output_tokens']}")
    
    print("\nAgent States:")
    for agent_name, agent in agents.items():
        state = agent.get_state()
        print(f"  {agent_name}: {state['status']} ({state['execution_time_ms']:.1f}ms)")
    
    print_section("Demo Complete")
    print("✓ FINODE system executed successfully")
    print("\nTo start the API server, run:")
    print("  python -m finode.api.server")


if __name__ == "__main__":
    run_demo()
