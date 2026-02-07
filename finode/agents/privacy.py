"""
PrivacyGovernorAgent

Applies differential privacy and enforces access control.
Simulates federated learning and privacy-preserving operations.
"""

import numpy as np
import time
import json
from typing import Dict, Any, List, Optional
from .base_agent import BaseAgent, AgentMessage, AgentState


class DifferentialPrivacyEngine:
    """
    Simple differential privacy implementation using Laplace mechanism.
    
    For numeric queries: answer + Lap(0, sensitivity/epsilon)
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 0.0001):
        """
        Initialize DP engine.
        
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Failure probability
        """
        self.epsilon = epsilon
        self.delta = delta
    
    def add_laplace_noise(self, value: float, sensitivity: float = 1.0) -> Tuple[float, float]:
        """
        Add Laplace noise to value.
        
        Args:
            value: Original numeric value
            sensitivity: Sensitivity of query (max change in output per record)
        
        Returns:
            (noisy_value, noise_amount)
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise, noise
    
    def add_gaussian_noise(self, value: float, sensitivity: float = 1.0) -> Tuple[float, float]:
        """
        Add Gaussian noise (alternative mechanism).
        
        Args:
            value: Original numeric value
            sensitivity: Sensitivity of query
        
        Returns:
            (noisy_value, noise_amount)
        """
        scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise = np.random.normal(0, scale)
        return value + noise, noise


class PrivacyGovernorAgent(BaseAgent):
    """
    Governs privacy and data access.
    
    Responsibilities:
    - Apply differential privacy noise
    - Enforce access control policies
    - Simulate federated learning
    - Privacy-preserving aggregation
    """
    
    def __init__(self, groq_client, epsilon: float = 1.0, delta: float = 0.0001):
        """
        Initialize PrivacyGovernorAgent.
        
        Args:
            groq_client: GroqClient instance
            epsilon: Differential privacy epsilon
            delta: Differential privacy delta
        """
        super().__init__("PrivacyGovernorAgent", groq_client)
        self.dp_engine = DifferentialPrivacyEngine(epsilon, delta)
        
        # Access control policy (user -> allowed_data_types)
        self.access_policy = {
            "analyst": ["aggregate", "trend", "public"],
            "researcher": ["aggregate", "trend", "public", "anonymized"],
            "admin": ["aggregate", "trend", "public", "anonymized", "detailed"],
        }
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Apply privacy transformations.
        
        Args:
            message: Message with content:
              - "user_role": User role for access control
              - "data": Data to protect
              - "operation": "aggregate", "filter", "noise", "federated"
        
        Returns:
            Response with privacy-protected data
        """
        self.state.status = "processing"
        start_time = time.time()
        
        try:
            user_role = message.content.get("user_role", "analyst")
            operation = message.content.get("operation", "aggregate")
            data = message.content.get("data", {})
            
            # Check access control
            if user_role not in self.access_policy:
                raise PermissionError(f"Unknown user role: {user_role}")
            
            allowed_types = self.access_policy[user_role]
            
            # Apply operation
            if operation == "aggregate":
                result = self._aggregate(data, allowed_types)
            elif operation == "filter":
                result = self._filter(data, allowed_types)
            elif operation == "noise":
                result = self._apply_noise(data)
            elif operation == "federated":
                result = self._federated_aggregation(data)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            self.state.status = "done"
            self.state.last_result = result
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            self.log_message(message)
            
            return AgentMessage(
                sender_agent="PrivacyGovernorAgent",
                recipient_agent=message.sender_agent,
                message_type="response",
                content=result,
            )
        
        except Exception as e:
            self.state.status = "error"
            self.state.error_message = str(e)
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            
            return AgentMessage(
                sender_agent="PrivacyGovernorAgent",
                recipient_agent=message.sender_agent,
                message_type="error",
                content={"error": str(e)},
            )
    
    def _aggregate(self, data: Dict[str, Any], allowed_types: List[str]) -> Dict[str, Any]:
        """Aggregate data with privacy"""
        output = {}
        
        for key, value in data.items():
            if key in allowed_types:
                output[key] = value
        
        return {
            "operation": "aggregate",
            "protected_data": output,
            "suppressed_fields": [k for k in data.keys() if k not in allowed_types],
            "privacy_level": "controlled_access",
        }
    
    def _filter(self, data: Dict[str, Any], allowed_types: List[str]) -> Dict[str, Any]:
        """Filter sensitive data"""
        filtered = {k: v for k, v in data.items() if k in allowed_types}
        
        return {
            "operation": "filter",
            "original_fields": len(data),
            "retained_fields": len(filtered),
            "data": filtered,
            "privacy_level": "filtered",
        }
    
    def _apply_noise(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply differential privacy noise to numeric fields"""
        noisy_data = {}
        noise_log = {}
        
        for key, value in data.items():
            if isinstance(value, (int, float)):
                noisy_value, noise_amount = self.dp_engine.add_laplace_noise(
                    float(value),
                    sensitivity=abs(value) * 0.1 if value != 0 else 1.0
                )
                noisy_data[key] = round(noisy_value, 2)
                noise_log[key] = round(noise_amount, 2)
            else:
                noisy_data[key] = value
        
        return {
            "operation": "apply_noise",
            "epsilon": self.dp_engine.epsilon,
            "delta": self.dp_engine.delta,
            "protected_data": noisy_data,
            "noise_amounts": noise_log,
            "privacy_level": "differentially_private",
        }
    
    def _federated_aggregation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate federated learning aggregation.
        In production, would aggregate updates from multiple clients.
        """
        # Simple average aggregation
        aggregated = {}
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, (int, float)):
                            if key not in aggregated:
                                aggregated[key] = []
                            aggregated[key].append(value)
            
            # Compute averages
            aggregated = {k: float(np.mean(v)) for k, v in aggregated.items()}
        
        return {
            "operation": "federated_aggregation",
            "num_clients": len(data) if isinstance(data, list) else 1,
            "aggregated_result": aggregated,
            "privacy_level": "federated",
        }


from typing import Tuple
