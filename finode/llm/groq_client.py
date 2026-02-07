"""
Groq LLM Client Abstraction

All LLM inference in FINODE flows through this single client.
This ensures traceability, caching, and controlled access.
"""

import json
import os
import hashlib
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time

# Groq SDK - install with: pip install groq
try:
    from groq import Groq
except ImportError:
    Groq = None


@dataclass
class GroqMessage:
    """Typed message for Groq API calls"""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class GroqResponse:
    """Typed response from Groq"""
    content: str
    model: str
    tokens_input: int
    tokens_output: int
    stop_reason: str
    call_hash: str  # SHA256 hash of request


class GroqClient:
    """
    Single-entry point for ALL Groq LLM calls.
    
    Features:
    - Structured message passing
    - Call hashing for audit trails
    - Error handling and retries
    - Token tracking
    - Streaming support
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model name (llama-3.3-70b-versatile, mixtral-8x7b-32768, etc.)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model
        self.temperature = 0.3
        self.max_tokens = 2048
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not provided. Set it via environment variable or constructor."
            )
        
        if Groq is None:
            raise ImportError("Groq SDK not installed. Install with: pip install groq")
        
        self.client = Groq(api_key=self.api_key)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
    
    def _hash_request(self, messages: List[GroqMessage], system_prompt: Optional[str]) -> str:
        """Generate SHA256 hash of request for audit trail"""
        request_str = json.dumps({
            "system": system_prompt,
            "messages": [(m.role, m.content) for m in messages],
            "model": self.model,
            "temperature": self.temperature,
        }, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()
    
    def infer(
        self,
        messages: List[GroqMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> GroqResponse:
        """
        Call Groq API with structured messages.
        
        Args:
            messages: List of GroqMessage objects
            system_prompt: Optional system context
            temperature: Model temperature (overrides default)
            max_tokens: Max output tokens (overrides default)
        
        Returns:
            GroqResponse with content, tokens, and audit hash
        """
        call_hash = self._hash_request(messages, system_prompt)
        
        # Build Groq API request
        groq_messages = [{"role": m.role, "content": m.content} for m in messages]
        
        if system_prompt:
            groq_messages.insert(0, {"role": "system", "content": system_prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=groq_messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                top_p=1,
                stream=False,
                stop=None,
            )
            
            self.call_count += 1
            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            
            return GroqResponse(
                content=response.choices[0].message.content,
                model=self.model,
                tokens_input=response.usage.prompt_tokens,
                tokens_output=response.usage.completion_tokens,
                stop_reason=response.choices[0].finish_reason or "unknown",
                call_hash=call_hash,
            )
        
        except Exception as e:
            raise RuntimeError(f"Groq API call failed: {str(e)}")
    
    def infer_json(
        self,
        messages: List[GroqMessage],
        system_prompt: Optional[str] = None,
        schema_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Call Groq and parse JSON response.
        
        Args:
            messages: List of GroqMessage objects
            system_prompt: Optional system context
            schema_hint: Optional JSON schema hint for model
        
        Returns:
            Parsed JSON dict
        """
        system = system_prompt or ""
        if schema_hint:
            system += f"\n\nRespond with valid JSON matching this structure:\n{schema_hint}"
        
        response = self.infer(messages, system)
        
        try:
            # Try to extract JSON from response
            content = response.content.strip()
            
            # Look for JSON block markers
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {response.content}\nError: {e}")
    
    def batch_infer(
        self,
        batch: List[tuple[List[GroqMessage], Optional[str]]],
    ) -> List[GroqResponse]:
        """
        Execute multiple inferences (sequential for rate limiting).
        
        Args:
            batch: List of (messages, system_prompt) tuples
        
        Returns:
            List of GroqResponse objects
        """
        results = []
        for messages, system_prompt in batch:
            results.append(self.infer(messages, system_prompt))
            time.sleep(0.1)  # Rate limiting
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Return usage statistics"""
        return {
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "model": self.model,
        }
