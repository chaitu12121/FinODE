"""
Base Agent Class

All agents inherit from BaseAgent and communicate via typed messages.
No direct state access between agents - only message passing.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class AgentMessage:
    """Typed message passed between agents"""
    sender_agent: str
    recipient_agent: str
    message_type: str  # "query", "response", "error", "ack"
    content: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    message_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender_agent,
            "recipient": self.recipient_agent,
            "type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "id": self.message_id,
        }


@dataclass
class AgentState:
    """Encapsulation of agent execution state"""
    agent_name: str
    status: str  # "idle", "processing", "done", "error"
    last_result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.agent_name,
            "status": self.status,
            "result": self.last_result,
            "error": self.error_message,
            "execution_time_ms": self.execution_time_ms,
        }


class BaseAgent:
    """
    Base class for all agents.
    
    Design principles:
    - Agents are single-responsibility
    - Communication via typed messages only
    - No shared state except through message passing
    - All decisions logged for auditability
    """
    
    def __init__(self, agent_name: str, groq_client=None):
        """
        Initialize base agent.
        
        Args:
            agent_name: Unique identifier for this agent
            groq_client: GroqClient instance (optional, not all agents need it)
        """
        self.agent_name = agent_name
        self.groq_client = groq_client
        self.state = AgentState(agent_name=agent_name, status="idle")
        self.message_history: List[AgentMessage] = []
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Process incoming message and return response.
        All subclasses must implement this.
        
        Args:
            message: Incoming AgentMessage
        
        Returns:
            Response AgentMessage
        """
        raise NotImplementedError(f"{self.agent_name} must implement process_message()")
    
    def log_message(self, message: AgentMessage) -> None:
        """Record message in history for audit trail"""
        self.message_history.append(message)
    
    def get_message_history(self) -> List[Dict[str, Any]]:
        """Return full message history"""
        return [msg.to_dict() for msg in self.message_history]
    
    def get_state(self) -> Dict[str, Any]:
        """Return current state (no internal state leakage)"""
        return self.state.to_dict()
