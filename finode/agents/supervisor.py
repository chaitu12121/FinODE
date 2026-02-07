"""
SupervisorAgent

Orchestrates multi-agent execution.
Enforces execution order, handles failures, and coordinates results.
"""

import time
import json
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentMessage, AgentState


class ExecutionDAG:
    """
    Directed Acyclic Graph (DAG) for task execution.
    Validates dependencies and determines execution order.
    """
    
    def __init__(self, tasks: List[Dict[str, Any]]):
        """
        Initialize execution DAG.
        
        Args:
            tasks: List of task definitions with id, dependencies, etc.
        """
        self.tasks = {t["task_id"]: t for t in tasks}
        self.execution_results = {}
        self._validate()
    
    def _validate(self) -> None:
        """Validate DAG (no cycles, all deps exist)"""
        for task_id, task in self.tasks.items():
            for dep_id in task.get("dependencies", []):
                if dep_id not in self.tasks:
                    raise ValueError(f"Task {task_id} depends on non-existent task {dep_id}")
    
    def get_executable_tasks(self) -> List[str]:
        """Get tasks ready to execute (all deps satisfied)"""
        executable = []
        for task_id, task in self.tasks.items():
            if task_id not in self.execution_results:
                deps = task.get("dependencies", [])
                if all(dep in self.execution_results for dep in deps):
                    executable.append(task_id)
        return executable
    
    def record_result(self, task_id: str, result: Dict[str, Any]) -> None:
        """Record task execution result"""
        self.execution_results[task_id] = result
    
    def is_complete(self) -> bool:
        """Check if all tasks have executed"""
        return len(self.execution_results) == len(self.tasks)


class SupervisorAgent(BaseAgent):
    """
    Orchestrates multi-agent execution.
    
    Responsibilities:
    - Plan validation
    - Execution scheduling
    - Agent coordination
    - Error handling and retries
    - Result aggregation
    """
    
    def __init__(self, groq_client, agents: Dict[str, BaseAgent]):
        """
        Initialize SupervisorAgent.
        
        Args:
            groq_client: GroqClient instance
            agents: Dict mapping agent name to agent instance
        """
        super().__init__("SupervisorAgent", groq_client)
        self.agents = agents
        self.max_retries = 3
        self.execution_log = []
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Execute a complete plan.
        
        Args:
            message: Message with content["plan"] (from PlannerAgent)
        
        Returns:
            Response with aggregated results
        """
        self.state.status = "processing"
        start_time = time.time()
        
        try:
            plan = message.content.get("plan", {})
            if not plan:
                raise ValueError("No plan provided")
            
            # Create execution DAG
            dag = ExecutionDAG(plan.get("tasks", []))
            
            # Execute tasks in dependency order
            execution_order = plan.get("execution_order", [])
            all_results = {}
            failed_tasks = []
            
            for task_id in execution_order:
                # Check if dependencies are satisfied
                task = dag.tasks[task_id]
                deps = task.get("dependencies", [])
                
                if not all(dep in all_results for dep in deps):
                    failed_tasks.append({
                        "task_id": task_id,
                        "reason": "Unsatisfied dependencies",
                    })
                    continue
                
                # Execute task with retries
                result = self._execute_task_with_retry(
                    task=task,
                    max_retries=self.max_retries,
                    previous_results=all_results
                )
                
                if result.get("success"):
                    all_results[task_id] = result
                    dag.record_result(task_id, result)
                else:
                    failed_tasks.append({
                        "task_id": task_id,
                        "reason": result.get("error", "Unknown error"),
                    })
            
            # Aggregate results
            response_content = {
                "plan_id": plan.get("plan_id", "unknown"),
                "execution_status": "success" if not failed_tasks else "partial",
                "tasks_executed": len(all_results),
                "tasks_failed": len(failed_tasks),
                "results": all_results,
                "failed_tasks": failed_tasks,
                "execution_order": execution_order,
            }
            
            self.state.status = "done" if not failed_tasks else "partial"
            self.state.last_result = response_content
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            self.log_message(message)
            
            return AgentMessage(
                sender_agent="SupervisorAgent",
                recipient_agent=message.sender_agent,
                message_type="response",
                content=response_content,
            )
        
        except Exception as e:
            self.state.status = "error"
            self.state.error_message = str(e)
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            
            return AgentMessage(
                sender_agent="SupervisorAgent",
                recipient_agent=message.sender_agent,
                message_type="error",
                content={"error": str(e)},
            )
    
    def _execute_task_with_retry(
        self,
        task: Dict[str, Any],
        max_retries: int = 3,
        previous_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute single task with retry logic.
        
        Args:
            task: Task definition
            max_retries: Number of retry attempts
            previous_results: Results from previous tasks
        
        Returns:
            Task execution result
        """
        agent_type = task.get("agent_type", "")
        task_id = task.get("task_id", "")
        parameters = task.get("parameters", {})
        
        # Find agent
        agent = None
        for agent_name, agent_instance in self.agents.items():
            if agent_type.lower() in agent_name.lower():
                agent = agent_instance
                break
        
        if not agent:
            return {
                "task_id": task_id,
                "success": False,
                "error": f"No agent found for type: {agent_type}"
            }
        
        # Try execution with retries
        for attempt in range(max_retries):
            try:
                # Build message for agent
                message = AgentMessage(
                    sender_agent="SupervisorAgent",
                    recipient_agent=agent.agent_name,
                    message_type="query",
                    content={
                        "task_id": task_id,
                        "description": task.get("description", ""),
                        "task_description": task.get("description", ""),
                        "parameters": parameters,
                        "previous_results": previous_results or {},
                    }
                )
                
                # Execute
                response = agent.process_message(message)
                
                if response.message_type == "error":
                    if attempt < max_retries - 1:
                        time.sleep(0.5)  # Brief wait before retry
                        continue
                    else:
                        return {
                            "task_id": task_id,
                            "success": False,
                            "error": response.content.get("error", "Unknown error"),
                            "attempts": attempt + 1,
                        }
                
                # Success
                return {
                    "task_id": task_id,
                    "success": True,
                    "agent": agent.agent_name,
                    "result": response.content,
                    "attempts": attempt + 1,
                    "execution_time_ms": agent.state.execution_time_ms,
                }
            
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                else:
                    return {
                        "task_id": task_id,
                        "success": False,
                        "error": str(e),
                        "attempts": attempt + 1,
                    }
        
        return {
            "task_id": task_id,
            "success": False,
            "error": "Max retries exceeded",
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of supervisor execution"""
        return {
            "agent_name": self.agent_name,
            "state": self.state.to_dict(),
            "agents_managed": list(self.agents.keys()),
            "execution_log": self.execution_log,
        }
