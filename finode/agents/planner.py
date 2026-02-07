"""
PlannerAgent

Converts user queries into structured JSON task graphs.
No tool access - only reasoning via Groq.
Output is an actionable multi-step plan.
"""

import json
import time
from typing import Dict, Any, Optional, List
from .base_agent import BaseAgent, AgentMessage, AgentState
from ..llm.groq_client import GroqMessage


class PlannerAgent(BaseAgent):
    """
    Translates natural language queries into structured execution plans.
    
    Responsibilities:
    - Parse user intent
    - Decompose into sub-tasks
    - Create DAG of dependencies
    - Assign agent responsibilities
    """
    
    def __init__(self, groq_client):
        """
        Initialize PlannerAgent.
        
        Args:
            groq_client: GroqClient instance for LLM calls
        """
        super().__init__("PlannerAgent", groq_client)
        self.system_prompt = """You are a planning agent for a financial intelligence system.
Your job is to take user queries and break them into structured execution plans.

For each query, produce a JSON response with this structure:
{
  "plan_id": "unique_id",
  "user_query": "original query",
  "intent": "what the user wants to achieve",
  "confidence": 0.95,
  "tasks": [
    {
      "task_id": "task_1",
      "description": "what to do",
      "agent_type": "RetrievalAgent|ExecutorAgent|TemporalReasoningAgent",
      "required_inputs": ["input1", "input2"],
      "outputs": ["output1"],
      "dependencies": ["task_0"],  // empty list if no deps
      "parameters": {}
    }
  ],
  "execution_order": ["task_1", "task_2"],
  "success_criteria": ["condition1", "condition2"]
}

Be precise and create minimal but complete plans.
"""
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Generate execution plan from user query.
        
        Args:
            message: Message with content["query"] (user query)
        
        Returns:
            Response with structured JSON plan
        """
        self.state.status = "processing"
        start_time = time.time()
        
        try:
            user_query = message.content.get("query", "")
            if not user_query:
                raise ValueError("No query provided")
            
            # Call Groq to generate plan
            groq_messages = [
                GroqMessage(role="user", content=f"Create a plan for: {user_query}")
            ]
            
            response = self.groq_client.infer_json(
                groq_messages,
                system_prompt=self.system_prompt
            )
            
            # Validate plan structure
            self._validate_plan(response)
            
            # Record state
            self.state.status = "done"
            self.state.last_result = response
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            
            # Log message
            self.log_message(message)
            
            # Return plan
            return AgentMessage(
                sender_agent="PlannerAgent",
                recipient_agent=message.sender_agent,
                message_type="response",
                content=response,
            )
        
        except Exception as e:
            self.state.status = "error"
            self.state.error_message = str(e)
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            
            return AgentMessage(
                sender_agent="PlannerAgent",
                recipient_agent=message.sender_agent,
                message_type="error",
                content={"error": str(e)},
            )
    
    def _validate_plan(self, plan: Dict[str, Any]) -> None:
        """Validate plan structure"""
        required_keys = ["plan_id", "user_query", "intent", "tasks", "execution_order"]
        for key in required_keys:
            if key not in plan:
                raise ValueError(f"Plan missing required key: {key}")
        
        if not isinstance(plan["tasks"], list) or len(plan["tasks"]) == 0:
            raise ValueError("Plan must have at least one task")
        
        for task in plan["tasks"]:
            required_task_keys = ["task_id", "description", "agent_type", "outputs", "dependencies"]
            for key in required_task_keys:
                if key not in task:
                    raise ValueError(f"Task missing required key: {key}")
