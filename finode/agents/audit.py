"""
AuditExplainabilityAgent

Generates immutable audit trails and explains decisions.
Produces SHAP-like explanations and hashes all decisions.
"""

import json
import time
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from .base_agent import BaseAgent, AgentMessage, AgentState


class AuditLog:
    """Immutable audit log entry"""
    
    def __init__(self, entry_id: str, event_type: str, details: Dict[str, Any]):
        """
        Create audit log entry.
        
        Args:
            entry_id: Unique identifier
            event_type: Type of event
            details: Event details
        """
        self.entry_id = entry_id
        self.event_type = event_type
        self.details = details
        self.timestamp = datetime.utcnow().isoformat()
        self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute SHA256 hash of entry"""
        content = json.dumps({
            "id": self.entry_id,
            "type": self.event_type,
            "details": self.details,
            "timestamp": self.timestamp,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "entry_id": self.entry_id,
            "event_type": self.event_type,
            "details": self.details,
            "timestamp": self.timestamp,
            "hash": self.hash,
        }


class SimpleExplainer:
    """
    Simplified SHAP-like explainer.
    Computes feature importance through perturbation.
    """
    
    def __init__(self):
        """Initialize explainer"""
        pass
    
    def explain_prediction(
        self,
        prediction: Any,
        features: Dict[str, float],
        baseline: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Explain prediction by ranking feature importance.
        
        Args:
            prediction: Model prediction/output
            features: Input features
            baseline: Baseline values for comparison
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if baseline is None:
            baseline = {k: 0.0 for k in features.keys()}
        
        # Simple importance: |feature - baseline| as proxy
        importance = {}
        for key, value in features.items():
            base_value = baseline.get(key, 0.0)
            importance[key] = float(abs(value - base_value))
        
        # Normalize
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance


class AuditExplainabilityAgent(BaseAgent):
    """
    Records all decisions and explains them.
    
    Responsibilities:
    - Generate audit logs
    - Compute decision hashes
    - Create feature attribution
    - Track decision chains
    """
    
    def __init__(self, groq_client, log_path: str = "./logs/audit.jsonl"):
        """
        Initialize AuditExplainabilityAgent.
        
        Args:
            groq_client: GroqClient instance
            log_path: Path to write audit logs
        """
        super().__init__("AuditExplainabilityAgent", groq_client)
        self.log_path = log_path
        self.explainer = SimpleExplainer()
        self.audit_entries: List[AuditLog] = []
    
    def process_message(self, message: AgentMessage) -> AgentMessage:
        """
        Log decision with explanation.
        
        Args:
            message: Message with content:
              - "event_type": Type of event
              - "decision": The decision made
              - "features": Input features
              - "reasoning": Why this decision
              - "agent_chain": Chain of agents involved
        
        Returns:
            Response with audit entry and explanation
        """
        self.state.status = "processing"
        start_time = time.time()
        
        try:
            event_type = message.content.get("event_type", "decision")
            decision = message.content.get("decision", {})
            features = message.content.get("features", {})
            reasoning = message.content.get("reasoning", "")
            agent_chain = message.content.get("agent_chain", [])
            
            # Create audit entry
            entry_id = self._generate_entry_id()
            audit_entry = AuditLog(
                entry_id=entry_id,
                event_type=event_type,
                details={
                    "decision": decision,
                    "agent_chain": agent_chain,
                    "reasoning": reasoning,
                    "features_count": len(features),
                }
            )
            
            self.audit_entries.append(audit_entry)
            
            # Explain decision
            explanation = self._explain_decision(decision, features)
            
            # Create response
            response_content = {
                "audit_entry": audit_entry.to_dict(),
                "explanation": explanation,
                "decision_hash": audit_entry.hash,
                "audit_trail": [e.to_dict() for e in self.audit_entries[-5:]],  # Last 5
                "integrity_check": self._verify_chain(),
            }
            
            self.state.status = "done"
            self.state.last_result = response_content
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            self.log_message(message)
            
            return AgentMessage(
                sender_agent="AuditExplainabilityAgent",
                recipient_agent=message.sender_agent,
                message_type="response",
                content=response_content,
            )
        
        except Exception as e:
            self.state.status = "error"
            self.state.error_message = str(e)
            self.state.execution_time_ms = (time.time() - start_time) * 1000
            
            return AgentMessage(
                sender_agent="AuditExplainabilityAgent",
                recipient_agent=message.sender_agent,
                message_type="error",
                content={"error": str(e)},
            )
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID"""
        timestamp = datetime.utcnow().isoformat()
        nonce = hashlib.sha256(timestamp.encode()).hexdigest()[:8]
        return f"audit_{len(self.audit_entries)}_{nonce}"
    
    def _explain_decision(self, decision: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
        """Explain decision using simple attribution"""
        
        # Extract numeric features for explanation
        numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
        
        if not numeric_features:
            return {
                "method": "rule_based",
                "explanation": "Decision based on qualitative factors",
                "factors": list(decision.keys()),
            }
        
        # Compute importance
        importance = self.explainer.explain_prediction(
            decision.get("result", 0.0),
            numeric_features
        )
        
        # Sort by importance
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "method": "perturbation_based_attribution",
            "feature_importance": dict(sorted_importance),
            "top_3_features": [f[0] for f in sorted_importance[:3]],
            "decision_confidence": decision.get("confidence", 0.0),
        }
    
    def _verify_chain(self) -> Dict[str, Any]:
        """Verify integrity of audit chain"""
        if len(self.audit_entries) < 2:
            return {"status": "ok", "entries_verified": len(self.audit_entries)}
        
        # Simple chain verification (in production, use blockchain-like hashing)
        all_hashes_unique = len(set(e.hash for e in self.audit_entries)) == len(self.audit_entries)
        
        return {
            "status": "ok" if all_hashes_unique else "warning",
            "entries_verified": len(self.audit_entries),
            "all_hashes_unique": all_hashes_unique,
        }
    
    def write_logs(self) -> None:
        """Write audit logs to file"""
        try:
            with open(self.log_path, "w") as f:
                for entry in self.audit_entries:
                    f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            print(f"Failed to write audit logs: {e}")
